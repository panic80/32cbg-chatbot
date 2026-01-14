import { useEffect, Dispatch, SetStateAction } from 'react';

interface KeyboardShortcutsOptions {
  showInlineCommand: boolean;
  selectedCommandIndex: number;
  inlineCommands: Array<{ command: string }>;
  setCommandOpen: Dispatch<SetStateAction<boolean>>;
  setSelectedCommandIndex: Dispatch<SetStateAction<number>>;
  setInput: (value: string) => void;
  setShowInlineCommand: Dispatch<SetStateAction<boolean>>;
  setShowHelpDialog?: Dispatch<SetStateAction<boolean>>;
}

export const useKeyboardShortcuts = ({
  showInlineCommand,
  selectedCommandIndex,
  inlineCommands,
  setCommandOpen,
  setSelectedCommandIndex,
  setInput,
  setShowInlineCommand,
  setShowHelpDialog,
}: KeyboardShortcutsOptions) => {
  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if (e.key === 'k' && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setCommandOpen((open) => !open);
      }
      // Shift+/ opens Help (common "?" shortcut)
      if (e.key === '/' && e.shiftKey && setShowHelpDialog) {
        e.preventDefault();
        setShowHelpDialog((prev) => !prev);
      }

      // Handle arrow navigation for inline commands
      if (showInlineCommand && inlineCommands.length > 0) {
        const total = inlineCommands.length;

        if (e.key === 'ArrowDown') {
          e.preventDefault();
          setSelectedCommandIndex((current) => (current + 1) % total);
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          setSelectedCommandIndex((current) => (current - 1 + total) % total);
        } else if (e.key === 'Enter') {
          e.preventDefault();
          const index = selectedCommandIndex % total;
          const selectedCommand = inlineCommands[index];
          if (selectedCommand) {
            setInput(selectedCommand.command + ' ');
          }
          setShowInlineCommand(false);
          // Focus is now handled inside ChatInput component
        } else if (e.key === 'Escape') {
          setShowInlineCommand(false);
        }
      } else if (
        showInlineCommand &&
        inlineCommands.length === 0 &&
        (e.key === 'ArrowDown' || e.key === 'ArrowUp')
      ) {
        e.preventDefault();
      }
    };

    document.addEventListener('keydown', down);
    return () => document.removeEventListener('keydown', down);
  }, [
    showInlineCommand,
    selectedCommandIndex,
    inlineCommands,
    setCommandOpen,
    setSelectedCommandIndex,
    setInput,
    setShowInlineCommand,
    setShowHelpDialog,
  ]);
};
