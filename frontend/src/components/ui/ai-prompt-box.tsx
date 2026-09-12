import { FormEvent, useEffect, useRef, useState } from "react";
import * as Dialog from "@radix-ui/react-dialog";
import * as Tooltip from "@radix-ui/react-tooltip";
import { AnimatePresence, motion } from "framer-motion";
import {
  Bot,
  BrainCircuit,
  ImagePlus,
  Mic,
  Paperclip,
  Search,
  Send,
  Square,
  WandSparkles,
  X
} from "lucide-react";
import { cn } from "../../lib/utils";

export type PromptMode = "search" | "think" | "canvas";

type AiPromptBoxProps = {
  disabled?: boolean;
  loading?: boolean;
  retrievalEnabled: boolean;
  mode: PromptMode;
  onModeChange: (mode: PromptMode) => void;
  onSubmit: (prompt: string) => void;
  onStop: () => void;
  onFilesSelected: (files: FileList) => void;
};

const modes: Array<{ id: PromptMode; label: string; icon: typeof Search }> = [
  { id: "search", label: "Search", icon: Search },
  { id: "think", label: "Think", icon: BrainCircuit },
  { id: "canvas", label: "Canvas", icon: WandSparkles }
];

export function AiPromptBox({
  disabled,
  loading,
  retrievalEnabled,
  mode,
  onModeChange,
  onSubmit,
  onStop,
  onFilesSelected
}: AiPromptBoxProps) {
  const [value, setValue] = useState("");
  const [recording, setRecording] = useState(false);
  const [seconds, setSeconds] = useState(0);
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!recording) {
      return;
    }

    const timer: ReturnType<typeof setInterval> = setInterval(() => {
      setSeconds((current) => current + 1);
    }, 1000);

    return () => clearInterval(timer);
  }, [recording]);

  function submit(event: FormEvent) {
    event.preventDefault();
    const prompt = value.trim();
    if (!prompt || disabled || loading) {
      return;
    }
    onSubmit(prompt);
    setValue("");
  }

  function toggleRecording() {
    setRecording((current) => !current);
    setSeconds(0);
  }

  return (
    <form className="prompt-shell" onSubmit={submit}>
      <div className="prompt-orbit">
        <Bot size={18} />
        <span>{retrievalEnabled ? "RAG active" : "Direct route"}</span>
      </div>

      <textarea
        value={value}
        disabled={disabled}
        onChange={(event) => setValue(event.target.value)}
        placeholder={
          retrievalEnabled
            ? "Ask against your indexed documents..."
            : "Ask SmartRoute to choose the best model path..."
        }
        rows={4}
        className="prompt-input"
      />

      <div className="prompt-actions">
        <Tooltip.Provider delayDuration={100}>
          <div className="mode-rail" aria-label="Prompt mode">
            {modes.map((item) => {
              const Icon = item.icon;
              return (
                <Tooltip.Root key={item.id}>
                  <Tooltip.Trigger asChild>
                    <button
                      type="button"
                      aria-label={item.label}
                      className={cn("icon-button", mode === item.id && "active")}
                      onClick={() => onModeChange(item.id)}
                    >
                      <Icon size={17} />
                    </button>
                  </Tooltip.Trigger>
                  <Tooltip.Portal>
                    <Tooltip.Content className="tooltip" sideOffset={8}>
                      {item.label}
                      <Tooltip.Arrow className="tooltip-arrow" />
                    </Tooltip.Content>
                  </Tooltip.Portal>
                </Tooltip.Root>
              );
            })}
          </div>

          <div className="prompt-tools">
            <input
              ref={fileRef}
              type="file"
              multiple
              accept=".pdf,.txt,.md"
              className="hidden"
              onChange={(event) => {
                if (event.target.files?.length) {
                  onFilesSelected(event.target.files);
                  event.target.value = "";
                }
              }}
            />

            <Tooltip.Root>
              <Tooltip.Trigger asChild>
                <button
                  type="button"
                  className="icon-button"
                  aria-label="Attach documents"
                  onClick={() => fileRef.current?.click()}
                >
                  <Paperclip size={17} />
                </button>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content className="tooltip" sideOffset={8}>
                  Attach PDF, TXT, or MD
                  <Tooltip.Arrow className="tooltip-arrow" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>

            <Dialog.Root>
              <Dialog.Trigger asChild>
                <button type="button" className="icon-button" aria-label="Image note">
                  <ImagePlus size={17} />
                </button>
              </Dialog.Trigger>
              <Dialog.Portal>
                <Dialog.Overlay className="dialog-overlay" />
                <Dialog.Content className="dialog-content">
                  <Dialog.Close className="dialog-close" aria-label="Close">
                    <X size={16} />
                  </Dialog.Close>
                  <Dialog.Title className="dialog-title">Image inputs are parked</Dialog.Title>
                  <Dialog.Description className="dialog-copy">
                    SmartRoute currently indexes PDF, TXT, and MD files. Image ingestion can be
                    added once the backend has a vision extraction path.
                  </Dialog.Description>
                </Dialog.Content>
              </Dialog.Portal>
            </Dialog.Root>

            <Tooltip.Root>
              <Tooltip.Trigger asChild>
                <button
                  type="button"
                  className={cn("icon-button", recording && "recording")}
                  aria-label="Toggle voice recorder"
                  onClick={toggleRecording}
                >
                  <Mic size={17} />
                </button>
              </Tooltip.Trigger>
              <Tooltip.Portal>
                <Tooltip.Content className="tooltip" sideOffset={8}>
                  {recording ? `Recording ${seconds}s` : "Voice note"}
                  <Tooltip.Arrow className="tooltip-arrow" />
                </Tooltip.Content>
              </Tooltip.Portal>
            </Tooltip.Root>

            <AnimatePresence mode="wait">
              {loading ? (
                <motion.button
                  key="stop"
                  type="button"
                  className="send-button stop"
                  onClick={onStop}
                  initial={{ scale: 0.92, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.92, opacity: 0 }}
                >
                  <Square size={16} />
                </motion.button>
              ) : (
                <motion.button
                  key="send"
                  type="submit"
                  className="send-button"
                  disabled={disabled || !value.trim()}
                  initial={{ scale: 0.92, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0.92, opacity: 0 }}
                >
                  <Send size={16} />
                </motion.button>
              )}
            </AnimatePresence>
          </div>
        </Tooltip.Provider>
      </div>
    </form>
  );
}
