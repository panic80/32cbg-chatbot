import type { Message } from '@/types';

type SerializableConversation = {
  conversationId: string;
  messages: Message[];
};

const buildFilename = (conversationId: string, extension: string) =>
  `conversation${conversationId ? `-${conversationId}` : ''}.${extension}`;

const triggerDownload = (blob: Blob, filename: string) => {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
};

const escapeCodeFence = (content: string) => content.replace(/```/g, '\\`\\`\\`');

export const exportConversationAsMarkdown = (messages: Message[], conversationId: string) => {
  const lines: string[] = [];
  lines.push(`# Conversation${conversationId ? ` ${conversationId}` : ''}`);
  lines.push('');

  messages.forEach((message) => {
    const role = message.sender === 'user' ? 'User' : 'Assistant';
    lines.push(`## ${role} (${new Date(message.timestamp).toLocaleString()})`);
    lines.push('');
    lines.push(escapeCodeFence(message.content));
    lines.push('');
  });

  const blob = new Blob([lines.join('\n')], { type: 'text/markdown' });
  triggerDownload(blob, buildFilename(conversationId, 'md'));
};

export const exportConversationAsJSON = (messages: Message[], conversationId: string) => {
  const payload: SerializableConversation = {
    conversationId,
    messages,
  };

  const blob = new Blob([JSON.stringify(payload, null, 2)], {
    type: 'application/json',
  });

  triggerDownload(blob, buildFilename(conversationId, 'json'));
};
