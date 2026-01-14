import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import { AlertCircle } from 'lucide-react';

interface DisclaimerModalProps {
  open: boolean;
  onAccept: () => void;
}

export function DisclaimerModal({ open, onAccept }: DisclaimerModalProps) {
  return (
    <AlertDialog open={open}>
      <AlertDialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto mx-4 md:mx-auto">
        <AlertDialogHeader>
          <AlertDialogTitle className="text-base md:text-xl text-amber-600 dark:text-amber-500 flex items-center gap-2 md:gap-3">
            <AlertCircle className="h-5 w-5 md:h-6 md:w-6 flex-shrink-0" />
            <span>IMPORTANT NOTICE - AI Assistant Disclaimer</span>
          </AlertDialogTitle>
          <AlertDialogDescription className="text-left space-y-3 md:space-y-4 pt-3 md:pt-4 text-sm md:text-base">
            <p className="font-medium">
              You are interacting with an AI-powered virtual assistant designed to help with
              Canadian Forces travel instructions and policies.
            </p>

            <p className="text-sm md:text-base">
              While this assistant strives to provide accurate and helpful information based on
              official policies and regulations, please be aware that:
            </p>

            <ul className="list-disc list-inside space-y-1 md:space-y-2 ml-3 md:ml-4 text-sm md:text-base">
              <li>AI-generated responses may contain errors or omissions</li>
              <li>Information may not reflect the most recent policy updates</li>
              <li>Complex situations may require human interpretation</li>
            </ul>

            <div className="bg-amber-50 dark:bg-amber-950/50 border border-amber-200 dark:border-amber-800 rounded-md p-3 md:p-4 mt-3 md:mt-4">
              <p className="font-bold text-amber-900 dark:text-amber-100 text-sm md:text-base">
                CRITICAL: For all travel claims, entitlements, and policy interpretations, you MUST
                verify the information with your Financial Services Administrator (FSA) or consult
                official documentation before taking any action.
              </p>
            </div>

            <p className="text-xs md:text-sm text-muted-foreground mt-3 md:mt-4">
              This AI assistant is a supplementary tool only and should not be your sole source for
              making financial or travel-related decisions.
            </p>
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter className="sticky bottom-0 bg-background pt-4 border-t mt-4">
          <AlertDialogAction
            onClick={onAccept}
            className="bg-primary hover:bg-primary/90 w-full md:w-auto"
          >
            I Understand and Accept
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  );
}
