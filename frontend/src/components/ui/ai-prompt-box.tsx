import React from "react";
import {
  ArrowUp,
  Paperclip,
  Square,
  X,
  FileText,
  Database,
  Zap,
  Scale,
  ShieldCheck,
  Loader2,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Button,
  Dialog,
  DialogContent,
  Textarea,
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/prompt-primitives";
import { cn } from "@/lib/classnames";

// ── Textarea ──────────────────────────────────────────────────────────────────

// ── Tooltip ───────────────────────────────────────────────────────────────────

// ── Dialog for Image Preview ──────────────────────────────────────────────────

// ── Button ────────────────────────────────────────────────────────────────────

// ── Prompt Input Primitives ───────────────────────────────────────────────────

interface PromptInputContextType {
  value: string;
  onValueChange: (value: string) => void;
  isLoading: boolean;
  onSubmit: () => void;
  disabled?: boolean;
}

const PromptInputContext = React.createContext<PromptInputContextType>({
  value: "",
  onValueChange: () => {},
  isLoading: false,
  onSubmit: () => {},
});

const usePromptInput = () => React.useContext(PromptInputContext);

interface PromptInputProps {
  value: string;
  onValueChange: (value: string) => void;
  isLoading?: boolean;
  onSubmit?: () => void;
  children: React.ReactNode;
  className?: string;
  disabled?: boolean;
  onDragOver?: (e: React.DragEvent) => void;
  onDragLeave?: (e: React.DragEvent) => void;
  onDrop?: (e: React.DragEvent) => void;
}

const PromptInput = React.forwardRef<HTMLDivElement, PromptInputProps>(
  (
    {
      value,
      onValueChange,
      isLoading = false,
      onSubmit = () => {},
      children,
      className,
      disabled = false,
      onDragOver,
      onDragLeave,
      onDrop,
    },
    ref
  ) => {
    return (
      <TooltipProvider>
        <PromptInputContext.Provider value={{ value, onValueChange, isLoading, onSubmit, disabled }}>
          <div
            ref={ref}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            className={cn(
              "rounded-3xl border bg-[#1F2023] p-3 shadow-2xl transition-all duration-200",
              className
            )}
          >
            {children}
          </div>
        </PromptInputContext.Provider>
      </TooltipProvider>
    );
  }
);
PromptInput.displayName = "PromptInput";

interface PromptInputTextareaProps {
  placeholder?: string;
  className?: string;
}

const PromptInputTextarea = React.forwardRef<HTMLTextAreaElement, PromptInputTextareaProps>(
  ({ placeholder = "Type a message...", className }, ref) => {
    const { value, onValueChange, onSubmit, disabled } = usePromptInput();
    const textareaRef = React.useRef<HTMLTextAreaElement>(null);

    React.useImperativeHandle(ref, () => textareaRef.current!);

    const adjustHeight = () => {
      const textarea = textareaRef.current;
      if (textarea) {
        textarea.style.height = "auto";
        textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
      }
    };

    React.useEffect(() => {
      adjustHeight();
    }, [value]);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        onSubmit();
      }
    };

    return (
      <Textarea
        ref={textareaRef}
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        className={cn("text-sm", className)}
      />
    );
  }
);
PromptInputTextarea.displayName = "PromptInputTextarea";

interface PromptInputActionProps {
  tooltip: string;
  children: React.ReactNode;
  className?: string;
  side?: "top" | "bottom" | "left" | "right";
  disabled?: boolean;
}

const PromptInputAction: React.FC<PromptInputActionProps> = ({
  tooltip,
  children,
  className,
  side = "top",
  disabled = false,
}) => {
  return (
    <Tooltip>
      <TooltipTrigger asChild disabled={disabled}>
        {children}
      </TooltipTrigger>
      <TooltipContent side={side} className={className}>
        {tooltip}
      </TooltipContent>
    </Tooltip>
  );
};

// ── Main PromptInputBox Component ─────────────────────────────────────────────

export interface PromptInputBoxProps {
  onSend?: (message: string, files?: File[]) => void;
  isLoading?: boolean;
  placeholder?: string;
  className?: string;
  strategy?: string;
  onStrategyChange?: (strategy: string) => void;
  ragEnabled?: boolean;
  onRagChange?: (enabled: boolean) => void;
  onUploadDocument?: (files: File[]) => void;
  isUploadingDoc?: boolean;
}

export const PromptInputBox = React.forwardRef((props: PromptInputBoxProps, ref: React.Ref<HTMLDivElement>) => {
  const {
    onSend = () => {},
    isLoading = false,
    placeholder = "Type your message here...",
    className,
    strategy = "cost_optimized",
    onStrategyChange,
    ragEnabled = false,
    onRagChange,
    onUploadDocument,
    isUploadingDoc = false,
  } = props;

  const [input, setInput] = React.useState("");
  const [files, setFiles] = React.useState<File[]>([]);
  const [filePreviews, setFilePreviews] = React.useState<{ [key: string]: string }>({});
  const [selectedImage, setSelectedImage] = React.useState<string | null>(null);
  const uploadInputRef = React.useRef<HTMLInputElement>(null);
  const promptBoxRef = React.useRef<HTMLDivElement>(null);

  const isImageFile = React.useCallback((file: File) => file.type.startsWith("image/"), []);
  const isDocFile = React.useCallback((file: File) => {
    const ext = file.name.split(".").pop()?.toLowerCase();
    return ["pdf", "txt", "md"].includes(ext || "");
  }, []);

  const processFile = React.useCallback((file: File) => {
    if (!ragEnabled) {
      alert("Please turn on the RAG toggle button to upload documents.");
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      console.warn("File too large (max 10MB)");
      return;
    }

    if (isImageFile(file)) {
      setFiles((prev) => [...prev, file]);
      const reader = new FileReader();
      reader.onload = (e) => {
        setFilePreviews((prev) => ({
          ...prev,
          [file.name]: (e.target?.result as string) || "",
        }));
      };
      reader.readAsDataURL(file);
    } else if (isDocFile(file)) {
      setFiles((prev) => [...prev, file]);
      onUploadDocument?.([file]);
    } else {
      setFiles((prev) => [...prev, file]);
    }
  }, [isDocFile, isImageFile, onUploadDocument, ragEnabled]);

  const handleDragOver = React.useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDragLeave = React.useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = React.useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      if (!ragEnabled) return;
      const droppedFiles = Array.from(e.dataTransfer.files);
      droppedFiles.forEach((file) => processFile(file));
    },
    [processFile, ragEnabled]
  );

  const handleRemoveFile = (index: number) => {
    const fileToRemove = files[index];
    if (fileToRemove && filePreviews[fileToRemove.name]) {
      setFilePreviews((prev) => {
        const next = { ...prev };
        delete next[fileToRemove.name];
        return next;
      });
    }
    setFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleSubmit = () => {
    if (input.trim() || files.length > 0) {
      onSend(input, files);
      setInput("");
      setFiles([]);
      setFilePreviews({});
    }
  };

  const hasContent = input.trim() !== "" || files.length > 0;

  return (
    <>
      <PromptInput
        value={input}
        onValueChange={setInput}
        isLoading={isLoading}
        onSubmit={handleSubmit}
        className={cn(
          "w-full bg-[#1F2023]/95 border-[#444444] shadow-[0_12px_40px_rgba(0,0,0,0.35)] transition-all duration-300 ease-in-out",
          className
        )}
        disabled={isLoading}
        ref={ref || promptBoxRef}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {/* File preview chips */}
        {files.length > 0 && (
          <div className="flex flex-wrap gap-2 p-1 pb-2 transition-all duration-300">
            {files.map((file, index) => (
              <div key={index} className="relative group">
                {isImageFile(file) ? (
                  <div
                    className="w-16 h-16 rounded-xl overflow-hidden cursor-pointer transition-all duration-300 border border-[#333333] hover:border-orange-400"
                    onClick={() => setSelectedImage(filePreviews[file.name] || "")}
                  >
                    <img
                      src={filePreviews[file.name] || ""}
                      alt={file.name}
                      className="h-full w-full object-cover"
                    />
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRemoveFile(index);
                      }}
                      className="absolute top-1 right-1 rounded-full bg-black/70 p-0.5 text-white hover:bg-black"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#2E3033] border border-white/10 text-xs text-gray-200">
                    <FileText className="h-4 w-4 text-orange-300" />
                    <span className="max-w-[140px] truncate">{file.name}</span>
                    <button
                      type="button"
                      onClick={() => handleRemoveFile(index)}
                      className="rounded-full hover:bg-black/50 p-0.5 text-gray-400 hover:text-white"
                    >
                      <X className="h-3 w-3" />
                    </button>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Text input */}
        <div className="transition-all duration-300">
          <PromptInputTextarea placeholder={placeholder} className="text-sm sm:text-base" />
        </div>

        {/* Bottom Actions Bar */}
        <div className="flex items-center justify-between gap-2 p-0 pt-2.5 border-t border-white/5 mt-1">
          {/* Left: RAG Toggle + (Upload Icon if RAG is ON) */}
          <div className="flex items-center gap-2">
            {/* RAG Toggle Button */}
            <button
              type="button"
              onClick={() => onRagChange?.(!ragEnabled)}
              className={cn(
                "flex items-center gap-1.5 px-2.5 py-1 rounded-xl border text-xs font-medium transition-all",
                ragEnabled
                  ? "bg-emerald-500/20 border-emerald-500/50 text-emerald-300 shadow-[0_0_12px_rgba(16,185,129,0.25)]"
                  : "bg-black/30 border-white/10 text-white/50 hover:text-white hover:bg-white/5"
              )}
              title={ragEnabled ? "RAG retrieval enabled (Knowledge base active)" : "Turn on RAG to enable document retrieval & uploads"}
            >
              <Database className="h-3.5 w-3.5" />
              <span>RAG</span>
              <span
                className={cn(
                  "h-1.5 w-1.5 rounded-full",
                  ragEnabled ? "bg-emerald-400 animate-pulse" : "bg-white/30"
                )}
              />
            </button>

            {/* Document Upload Icon: ONLY visible when ragEnabled is true */}
            <AnimatePresence>
              {ragEnabled && (
                <motion.div
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.8 }}
                  transition={{ duration: 0.15 }}
                >
                  <PromptInputAction tooltip="Upload document to knowledge base (.pdf, .txt, .md)">
                    <button
                      type="button"
                      onClick={() => uploadInputRef.current?.click()}
                      disabled={isUploadingDoc}
                      className={cn(
                        "flex h-8 items-center gap-1.5 px-2.5 text-xs rounded-xl bg-white/10 hover:bg-white/20 border border-white/10 text-white transition-all",
                        isUploadingDoc && "opacity-75 cursor-wait"
                      )}
                    >
                      {isUploadingDoc ? (
                        <Loader2 className="h-3.5 w-3.5 animate-spin text-orange-300" />
                      ) : (
                        <Paperclip className="h-3.5 w-3.5 text-orange-300" />
                      )}
                      <span className="hidden sm:inline">Upload Files</span>
                      <input
                        ref={uploadInputRef}
                        type="file"
                        multiple
                        className="hidden"
                        onChange={(e) => {
                          if (e.target.files && e.target.files.length > 0) {
                            Array.from(e.target.files).forEach((f) => processFile(f));
                          }
                          if (e.target) e.target.value = "";
                        }}
                        accept=".pdf,.txt,.md"
                      />
                    </button>
                  </PromptInputAction>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right: Strategy Selector Pills + Clean Send Button */}
          <div className="flex items-center gap-2">
            {/* Strategy Pills */}
            <div className="flex items-center bg-black/40 border border-white/10 rounded-xl p-0.5 text-xs">
              <button
                type="button"
                onClick={() => onStrategyChange?.("cost_optimized")}
                className={cn(
                  "flex items-center gap-1 px-2 sm:px-2.5 py-1 rounded-lg transition-all text-xs",
                  strategy === "cost_optimized"
                    ? "bg-white/20 text-white font-medium shadow-sm"
                    : "text-white/50 hover:text-white/80"
                )}
                title="Smallest configured capable model, lowest cost"
              >
                <Zap className="h-3.5 w-3.5 text-emerald-400" />
                <span className="hidden sm:inline">Cost Optimized</span>
                <span className="sm:hidden text-[10px]">Cost</span>
              </button>

              <button
                type="button"
                onClick={() => onStrategyChange?.("balanced")}
                className={cn(
                  "flex items-center gap-1 px-2 sm:px-2.5 py-1 rounded-lg transition-all text-xs",
                  strategy === "balanced"
                    ? "bg-white/20 text-white font-medium shadow-sm"
                    : "text-white/50 hover:text-white/80"
                )}
                title="Balanced quality and cost"
              >
                <Scale className="h-3.5 w-3.5 text-blue-400" />
                <span className="hidden sm:inline">Balanced</span>
                <span className="sm:hidden text-[10px]">Bal</span>
              </button>

              <button
                type="button"
                onClick={() => onStrategyChange?.("quality_first")}
                className={cn(
                  "flex items-center gap-1 px-2 sm:px-2.5 py-1 rounded-lg transition-all text-xs",
                  strategy === "quality_first"
                    ? "bg-white/20 text-white font-medium shadow-sm"
                    : "text-white/50 hover:text-white/80"
                )}
                title="Prefer the highest-quality configured route"
              >
                <ShieldCheck className="h-3.5 w-3.5 text-purple-400" />
                <span className="hidden sm:inline">Quality First</span>
                <span className="sm:hidden text-[10px]">Quality</span>
              </button>
            </div>

            {/* Clean Send Button (NO microphone / speaker icon) */}
            <PromptInputAction
              tooltip={isLoading ? "Stop generation" : hasContent ? "Send message" : "Type a message to send"}
            >
              <Button
                type="button"
                variant="default"
                size="icon"
                className={cn(
                  "h-8 w-8 rounded-full transition-all duration-200",
                  isLoading
                    ? "bg-red-500/80 hover:bg-red-500 text-white"
                    : hasContent
                    ? "bg-white hover:bg-white/90 text-[#1F2023] shadow-md cursor-pointer"
                    : "bg-white/10 text-white/30 cursor-not-allowed"
                )}
                onClick={() => {
                  if (hasContent && !isLoading) handleSubmit();
                }}
                disabled={isLoading || !hasContent}
              >
                {isLoading ? (
                  <Square className="h-3.5 w-3.5 fill-white animate-pulse" />
                ) : (
                  <ArrowUp className="h-4 w-4" />
                )}
              </Button>
            </PromptInputAction>
          </div>
        </div>
      </PromptInput>

      {/* Image Preview Modal */}
      <Dialog open={!!selectedImage} onOpenChange={(open) => !open && setSelectedImage(null)}>
        <DialogContent className="max-w-3xl overflow-hidden p-2 bg-black/90 border border-white/20">
          {selectedImage && (
            <img src={selectedImage} alt="Preview" className="w-full max-h-[80vh] object-contain rounded-xl" />
          )}
        </DialogContent>
      </Dialog>
    </>
  );
});
PromptInputBox.displayName = "PromptInputBox";
