import { TooltipProvider } from '@/components/ui/tooltip';
import { DisclaimerModal } from '@/components/DisclaimerModal';
import { BackgroundEffects } from './ChatPage/components/BackgroundEffects';
import { ChatHeader } from './ChatPage/components/ChatHeader';
import { ChatInput } from './ChatPage/components/ChatInput';
import { HelpDialog } from './ChatPage/components/HelpDialog';
import { useChatController } from './ChatPage/hooks';
import { ChatCommandPalette } from './ChatPage/components/ChatCommandPalette';
import { ChatMessagesPanel } from './ChatPage/components/ChatMessagesPanel';
import { MaintenanceBanner } from './ChatPage/components/MaintenanceBanner';
import { MAINTENANCE_MODE } from '@/constants';

interface ChatPageProps {
  theme?: string;
  toggleTheme?: () => void;
}

/**
 * Enhanced Chat page with modern UI/UX improvements
 */
const ChatPage: React.FC<ChatPageProps> = ({ theme: propTheme, toggleTheme: propToggleTheme }) => {
  const {
    commandPaletteProps,
    chatHeaderProps,
    messagesPanelProps,
    chatInputProps,
    helpDialogProps,
    disclaimerProps,
  } = useChatController({ propTheme, propToggleTheme });

  return (
    <TooltipProvider>
      {/* Disclaimer Modal */}
      <DisclaimerModal {...disclaimerProps} />

      <div className="flex h-screen bg-[var(--background)] text-[var(--text)] relative overflow-x-hidden overflow-y-hidden">
        {/* Static Background Elements (motion removed to fix flickering) */}
        <BackgroundEffects />

        {/* Command Palette */}
        <ChatCommandPalette {...commandPaletteProps} />

        {/* Main Content - Full Width */}
        <div className="flex-1 flex flex-col relative w-full">
          {/* Enhanced Header */}
          <ChatHeader {...chatHeaderProps} />

          {/* Maintenance Banner */}
          {MAINTENANCE_MODE && <MaintenanceBanner />}

          <ChatMessagesPanel {...messagesPanelProps} />

          {/* Enhanced Input Area */}
          <ChatInput {...chatInputProps} maintenanceMode={MAINTENANCE_MODE} />
        </div>
      </div>

      {/* Help Dialog */}
      <HelpDialog {...helpDialogProps} />
    </TooltipProvider>
  );
};

export default ChatPage;
