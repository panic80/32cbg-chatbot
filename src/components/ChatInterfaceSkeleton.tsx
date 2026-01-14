import React from 'react';
import { Skeleton, SkeletonAvatar, SkeletonChatMessage } from './ui/skeleton';

export const ChatInterfaceSkeleton: React.FC = () => {
  return (
    <div className="chat-interface">
      {/* Header Skeleton */}
      <div className="chat-header">
        <div className="header-content">
          <div className="flex items-center gap-3">
            <SkeletonAvatar size="default" />
            <div className="space-y-2">
              <Skeleton className="h-5 w-32" />
              <Skeleton className="h-3 w-20 opacity-50" />
            </div>
          </div>
        </div>
      </div>

      {/* Messages Container Skeleton */}
      <div className="messages-container">
        <div className="space-y-6 p-4">
          {/* User message skeleton */}
          <SkeletonChatMessage variant="sent" />

          {/* Assistant message skeleton */}
          <SkeletonChatMessage variant="received" />

          {/* Another user message */}
          <SkeletonChatMessage variant="sent" />

          {/* Loading assistant message */}
          <div className="flex items-start gap-3">
            <SkeletonAvatar size="sm" />
            <div className="bg-muted rounded-lg p-3">
              <div className="loading-dots">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Input Area Skeleton */}
      <div className="input-container">
        <div className="input-wrapper">
          <div className="input-field">
            <Skeleton className="h-12 w-full rounded-md" />
          </div>
        </div>
        <div className="powered-by-footer">
          <Skeleton className="h-3 w-32 mx-auto" />
        </div>
      </div>
    </div>
  );
};

export default ChatInterfaceSkeleton;
