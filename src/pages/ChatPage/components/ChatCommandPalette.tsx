import React from 'react';
import { Sparkles } from 'lucide-react';
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from '@/components/ui/command';

interface ChatCommandPaletteProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onCommandSelect: (value: string) => void;
}

export const ChatCommandPalette: React.FC<ChatCommandPaletteProps> = ({
  open,
  onOpenChange,
  onCommandSelect,
}) => (
  <CommandDialog open={open} onOpenChange={onOpenChange}>
    <CommandInput placeholder="Type a command or search..." />
    <CommandList>
      <CommandEmpty>No results found.</CommandEmpty>
      <CommandGroup heading="Actions">
        <CommandItem onSelect={() => onCommandSelect('TD claim requirements')}>
          <Sparkles className="mr-2 h-4 w-4" />
          TD claim requirements
        </CommandItem>
        <CommandItem onSelect={() => onCommandSelect('LTA eligibility')}>
          <Sparkles className="mr-2 h-4 w-4" />
          LTA eligibility
        </CommandItem>
        <CommandItem onSelect={() => onCommandSelect('Travel authorization')}>
          <Sparkles className="mr-2 h-4 w-4" />
          Travel authorization
        </CommandItem>
      </CommandGroup>
    </CommandList>
  </CommandDialog>
);

export default ChatCommandPalette;
