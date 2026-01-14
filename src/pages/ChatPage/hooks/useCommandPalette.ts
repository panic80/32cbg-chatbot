import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ChangeEvent, KeyboardEvent, Dispatch, SetStateAction } from 'react';
import { INLINE_COMMANDS } from '../constants/commands';
import { useKeyboardShortcuts } from './useKeyboardShortcuts';

interface UseCommandPaletteOptions {
  setInput: (value: string) => void;
  onSubmit: () => void | Promise<void>;
  setShowHelpDialog?: Dispatch<SetStateAction<boolean>>;
}

export const useCommandPalette = ({
  setInput,
  onSubmit,
  setShowHelpDialog,
}: UseCommandPaletteOptions) => {
  const [commandOpen, setCommandOpen] = useState(false);
  const [showInlineCommand, setShowInlineCommand] = useState(false);
  const [commandFilter, setCommandFilter] = useState('');
  const [selectedCommandIndex, setSelectedCommandIndex] = useState(0);

  const commands = INLINE_COMMANDS;

  const filteredCommands = useMemo(() => {
    if (!commandFilter) {
      return commands;
    }
    const lowerFilter = commandFilter.toLowerCase();
    return commands.filter((cmd) => cmd.command.toLowerCase().startsWith(lowerFilter));
  }, [commandFilter, commands]);

  const handleInputChange = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const value = event.target.value;
      setInput(value);

      if (value.startsWith('/') && value.length > 1) {
        const lower = value.toLowerCase();
        const matches = commands.filter((cmd) => cmd.command.toLowerCase().startsWith(lower));
        setShowInlineCommand(matches.length > 0);
        setCommandFilter(lower);
        setSelectedCommandIndex(0);
      } else {
        setShowInlineCommand(false);
        setCommandFilter('');
        setSelectedCommandIndex(0);
      }
    },
    [commands, setInput],
  );

  const handleKeyPress = useCallback(
    (event: KeyboardEvent<HTMLInputElement>) => {
      if (event.key === 'Enter' && !event.shiftKey && !showInlineCommand) {
        event.preventDefault();
        onSubmit();
      }
    },
    [onSubmit, showInlineCommand],
  );

  useEffect(() => {
    if (!showInlineCommand) {
      setCommandFilter('');
      setSelectedCommandIndex(0);
    }
  }, [showInlineCommand]);

  useKeyboardShortcuts({
    showInlineCommand,
    selectedCommandIndex,
    inlineCommands: filteredCommands,
    setCommandOpen,
    setSelectedCommandIndex,
    setInput,
    setShowInlineCommand,
    setShowHelpDialog,
  });

  return {
    commandOpen,
    setCommandOpen,
    showInlineCommand,
    setShowInlineCommand,
    selectedCommandIndex,
    setSelectedCommandIndex,
    handleInputChange,
    handleKeyPress,
    commands: filteredCommands,
  };
};
