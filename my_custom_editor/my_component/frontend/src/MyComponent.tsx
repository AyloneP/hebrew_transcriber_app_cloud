import {
  Streamlit,
  withStreamlitConnection,
  ComponentProps,
} from "streamlit-component-lib"
import React, { useEffect, useRef, useState, useMemo } from "react"

interface WordData {
  id: number
  word: string
  prefix_punc?: string
  punctuation?: string
  clean_word?: string
  start: number
  end: number
  confidence: number
  speaker: string
  deleted?: boolean
}

const getWordColor = (confidence: number, isDark: boolean): string => {
  if (confidence < 0.5) return isDark ? "#7f1d1d" : "#ff4b4b" 
  if (confidence <= 0.9) return isDark ? "#713f12" : "#ffe14b" 
  return isDark ? "#374151" : "#f0f2f6" 
}

const FILLERS = new Set(["אה", "אממ", "אהה", "אמ", "um", "uh", "mhm", "mm", "ah", "er", "hmm"])

const formatTime = (seconds: number) => {
  const m = Math.floor(seconds / 60)
  const s = Math.floor(seconds % 60)
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`
}

const MyComponent: React.FC<ComponentProps> = (props) => {
  const wordsData: WordData[] = props.args["words_data"] || []
  const speakerNames: Record<string, string> = props.args["speaker_names"] || {}
  const gapThreshold: number = props.args["gap_threshold"] || 0.6
  const searchQuery: string = props.args["search_query"] || ""
  const audioSrc: string = props.args["audio_src"] || "" 
  const playbackRate: number = props.args["playback_rate"] || 1.0

  const [activeWords, setActiveWords] = useState<WordData[]>([])
  const [currentTime, setCurrentTime] = useState<number>(0)
  
  const isDark = props.theme?.base === "dark"
  
  const containerRef = useRef<HTMLDivElement>(null)
  const scrollPosRef = useRef<number>(0)
  
  // משתנים למעקב אחרי הגלילה האוטומטית
  const isUserScrollingRef = useRef<boolean>(false)
  const scrollTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const lastScrolledIdRef = useRef<number | null>(null)

  const handleScroll = (e: React.UIEvent<HTMLDivElement>) => {
    scrollPosRef.current = e.currentTarget.scrollTop
    // זיהוי שהמשתמש גולל ידנית והשהיית הגלילה האוטומטית
    isUserScrollingRef.current = true
    if (scrollTimeoutRef.current) clearTimeout(scrollTimeoutRef.current)
    scrollTimeoutRef.current = setTimeout(() => {
      isUserScrollingRef.current = false
    }, 2000)
  }

  const getParentAudio = () => {
    try {
      const audios = window.parent.document.querySelectorAll('audio');
      for (let i = 0; i < audios.length; i++) {
        if (!audios[i].src.startsWith('data:')) {
          return audios[i];
        }
      }
      return null;
    } catch (e) {
      return null;
    }
  };

  useEffect(() => {
    const audio = getParentAudio();
    if (audio) {
      audio.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const isTyping = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;
      
      // מקש רווח (Space) לעצירה/ניגון
      if (e.code === 'Space' && !isTyping) {
        e.preventDefault();
        const audio = getParentAudio();
        if (audio) {
          if (audio.paused) audio.play();
          else audio.pause();
        }
        return;
      }

      // קיצורים - ביטול (Ctrl+Z) או שחזור (Ctrl+Y / Ctrl+Shift+Z)
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') {
        if (!isTyping) e.preventDefault(); // מונע התנגשות רק אם לא מקלידים טקסט
        if (e.shiftKey) {
          Streamlit.setComponentValue({ action: "redo", ts: Date.now() });
        } else {
          Streamlit.setComponentValue({ action: "undo", ts: Date.now() });
        }
      } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') {
        if (!isTyping) e.preventDefault();
        Streamlit.setComponentValue({ action: "redo", ts: Date.now() });
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    try { window.parent.document.addEventListener('keydown', handleKeyDown); } catch(e){}
    
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
      try { window.parent.document.removeEventListener('keydown', handleKeyDown); } catch(e){}
    };
  }, []);

  useEffect(() => {
    let animationFrameId: number;
    const trackAudio = () => {
      const audio = getParentAudio();
      if (audio) {
        setCurrentTime(audio.currentTime);
      }
      animationFrameId = requestAnimationFrame(trackAudio);
    };
    trackAudio();
    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  // מנגנון סנכרון וגלילה אוטומטית (Auto-Scroll)
  useEffect(() => {
    if (!isUserScrollingRef.current && containerRef.current) {
      const activeEl = containerRef.current.querySelector('.active-word');
      if (activeEl && activeEl.id !== `word-${lastScrolledIdRef.current}`) {
        lastScrolledIdRef.current = Number(activeEl.id.replace('word-', ''));
        activeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [currentTime]);

  useEffect(() => {
    setActiveWords(wordsData.filter((w) => !w.deleted))
    
    const restoreScroll = () => {
      if (containerRef.current) {
        containerRef.current.scrollTop = scrollPosRef.current
      }
    }
    restoreScroll();
    setTimeout(restoreScroll, 10);
    setTimeout(restoreScroll, 50);
  }, [wordsData])

  useEffect(() => {
    Streamlit.setFrameHeight()
  })

  const handleSeek = (time: number) => {
    const audio = getParentAudio();
    if (audio) {
      audio.currentTime = time;
      audio.play();
    }
  }

  const speakerBlocks = useMemo(() => {
    const blocks: { speaker: string; words: WordData[] }[] = []
    let currentBlock: { speaker: string; words: WordData[] } | null = null

    activeWords.forEach((w) => {
      if (!currentBlock || currentBlock.speaker !== w.speaker) {
        currentBlock = { speaker: w.speaker, words: [] }
        blocks.push(currentBlock)
      }
      currentBlock.words.push(w)
    })
    return blocks
  }, [activeWords])

  const handleParagraphBlur = (
    e: React.FocusEvent<HTMLParagraphElement>,
    originalBlockWords: WordData[],
    speaker: string
  ) => {
    const newText = e.currentTarget.innerText.trim()
    const newWordsArr = newText.split(/\s+/).filter(Boolean)
    if (newWordsArr.length === 0) return

    const startTime = originalBlockWords[0]?.start ?? 0
    const endTime = originalBlockWords[originalBlockWords.length - 1]?.end ?? startTime + 1
    const totalDuration = Math.max(endTime - startTime, 0.1)
    const chunkTime = totalDuration / newWordsArr.length

    const startIndex = wordsData.findIndex((w) => w.id === originalBlockWords[0]?.id)
    const endIndex = wordsData.findIndex(
      (w) => w.id === originalBlockWords[originalBlockWords.length - 1]?.id
    )
    if (startIndex === -1 || endIndex === -1) return

    const maxId = wordsData.reduce((acc, curr) => Math.max(acc, curr.id), 0)
    
    // Regex לחילוץ סימני פיסוק מההתחלה והסוף של המילה בעריכה ידנית
    const PUNC_REGEX = /^([.,?!״׳"()[\]{}:;\-]*)(.*?)([.,?!״׳"()[\]{}:;\-]*)$/;

    const updatedBlockWords: WordData[] = newWordsArr.map((w, i) => {
      const match = w.match(PUNC_REGEX);
      const prefix = match ? match[1] : "";
      const coreWord = match && match[2] ? match[2] : w;
      const suffix = match ? match[3] : "";

      return {
        id: maxId + 1 + i,
        word: coreWord,
        prefix_punc: prefix,
        punctuation: suffix,
        clean_word: coreWord.toLowerCase(),
        start: startTime + i * chunkTime,
        end: startTime + (i + 1) * chunkTime,
        confidence: 1.0,
        speaker: speaker,
        deleted: false,
      }
    })

    const updatedFullList = [
      ...wordsData.slice(0, startIndex),
      ...updatedBlockWords,
      ...wordsData.slice(endIndex + 1),
    ]
    
    Streamlit.setComponentValue({ action: "update", data: updatedFullList, ts: Date.now() })
  }

  const handleWordDoubleClick = (e: React.MouseEvent, id: number) => {
    e.preventDefault()
    // מניעת הסימון הכחול של הטקסט על ידי הדפדפן
    if (window.getSelection) {
      window.getSelection()?.removeAllRanges();
    }
    Streamlit.setComponentValue({ action: "select", word_id: id, ts: Date.now() })
  }

  const themeBg = isDark ? "#1e1e1e" : "#ffffff"
  const themeText = isDark ? "#e0e0e0" : "#000000"
  const themeBorder = isDark ? "#333333" : "#e0e0e0"
  const themeSpeaker = isDark ? "#90caf9" : "#1976d2"

  return (
    <div
      ref={containerRef}
      onScroll={handleScroll}
      dir="rtl"
      style={{
        position: "relative",
        fontFamily: "system-ui, -apple-system, sans-serif",
        fontSize: "18px",
        lineHeight: "2.2",
        backgroundColor: themeBg,
        color: themeText,
        padding: "30px",
        borderRadius: "12px",
        border: `1px solid ${themeBorder}`,
        boxShadow: "0 4px 6px rgba(0,0,0,0.05)",
        maxHeight: "750px", 
        overflowY: "auto",
        overflowX: "hidden",
        transition: "all 0.3s ease",
        scrollBehavior: "smooth"
      }}
    >
      {speakerBlocks.map((block, bIdx) => {
        const spkName = speakerNames[block.speaker] || `דובר ${block.speaker}`
        const blockKey = `block-${bIdx}-${block.words.map(w => w.id).join('-')}`
        const blockStartTime = block.words.length > 0 ? block.words[0].start : 0
        const timeString = formatTime(blockStartTime)
        
        return (
          <div key={bIdx} style={{ marginBottom: "25px" }}>
            <div
              style={{
                fontWeight: "bold",
                marginBottom: "8px",
                userSelect: "none",
                fontSize: "16px",
                display: "flex",
                alignItems: "center",
                gap: "8px"
              }}
            >
              <span 
                onClick={() => handleSeek(blockStartTime)}
                title="לחץ כדי לקפוץ לנקודה זו באודיו"
                style={{ color: "#78909c", fontSize: "14px", fontWeight: "normal", cursor: "pointer", textDecoration: "underline" }}
              >
                [{timeString}]
              </span>
              <span style={{ color: themeSpeaker }}>
                [{spkName}]:
              </span>
            </div>
            <p
              key={blockKey}
              contentEditable
              suppressContentEditableWarning
              onBlur={(e) => handleParagraphBlur(e, block.words, block.speaker)}
              style={{
                outline: "none",
                minHeight: "30px",
                padding: "10px",
                borderRadius: "8px",
                border: "1px solid transparent",
                margin: "0",
                transition: "border 0.2s",
                wordBreak: "break-word" 
              }}
              onFocus={(e) => e.currentTarget.style.border = `1px dashed ${themeSpeaker}`}
            >
              {block.words.map((w, wIdx) => {
                const isFiller = FILLERS.has((w.clean_word || "").toLowerCase())
                const wordWithPunc = `${w.prefix_punc || ""}${w.word}${w.punctuation || ""}`
                const display = isFiller ? `[${wordWithPunc}]` : wordWithPunc
                
                const isMatch = searchQuery && w.word.toLowerCase().includes(searchQuery.toLowerCase())
                const isPlaying = currentTime >= w.start && currentTime < w.end
                
                const baseColor = getWordColor(w.confidence, isDark)
                const bgColor = isMatch ? (isDark ? "#00b248" : "#00e676") : baseColor
                const fontWeightStyle = isMatch ? "bold" : "normal"
                
                const highlightShadow = isPlaying ? (isDark ? "0 0 0 2px #64b5f6" : "0 0 0 2px #2196F3") : "none"

                let showGap = false
                let gapTime = 0
                if (wIdx < block.words.length - 1) {
                  const nextW = block.words[wIdx + 1]
                  gapTime = nextW.start - w.end
                  if (gapTime >= gapThreshold) showGap = true
                }

                return (
                  <React.Fragment key={w.id}>
                    <span
                      id={`word-${w.id}`}
                      className={isPlaying ? "active-word" : ""}
                      onDoubleClick={(e) => handleWordDoubleClick(e, w.id)}
                      title="לחץ פעמיים לשמיעת האודיו ושינוי דובר"
                      style={{
                        backgroundColor: bgColor,
                        fontWeight: fontWeightStyle,
                        boxShadow: highlightShadow,
                        color: isMatch && !isDark ? "#000" : "inherit",
                        padding: "2px 6px",
                        borderRadius: "4px",
                        opacity: isFiller ? 0.6 : 1,
                        fontStyle: isFiller ? "italic" : "normal",
                        cursor: "pointer",
                        transition: "box-shadow 0.1s"
                      }}
                    >
                      {display}
                    </span>
                    
                    <span style={{ whiteSpace: "pre-wrap" }}> </span>
                    
                    {showGap && (
                      <span
                        style={{
                          color: "#ff9800",
                          fontSize: "13px",
                          userSelect: "none",
                        }}
                        title={`שתיקה של ${gapTime.toFixed(1)} שניות`}
                      >
                        [⏳]
                      </span>
                    )}
                    
                    {showGap && <span style={{ whiteSpace: "pre-wrap" }}> </span>}
                    
                  </React.Fragment>
                )
              })}
            </p>
          </div>
        )
      })}
    </div>
  )
}

export default withStreamlitConnection(MyComponent)