import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FileText,
  FolderOpen,
  GraduationCap,
  Wrench,
  ChevronRight,
  ExternalLink,
  Sparkles,
  Clock,
  Search,
  Command,
  CheckCircle,
  TrendingUp,
  Users,
  Files,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

// Stats Card Component
const StatsCard = ({ stat, delay }) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.3 }}
      className={cn(
        'relative overflow-hidden rounded-xl',
        'bg-[var(--card)]/50 backdrop-blur-sm',
        'border border-[var(--border)]/50',
        'p-6 group hover:shadow-lg transition-all duration-300',
      )}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-[var(--text-secondary)]">{stat.label}</p>
          <p className="text-3xl font-semibold mt-2 text-[var(--text)]">{stat.value}</p>
          <div className="flex items-center mt-2 gap-1">
            <stat.trendIcon
              className={cn('w-4 h-4', stat.trend === 'up' ? 'text-green-500' : 'text-red-500')}
            />
            <span
              className={cn(
                'text-sm font-medium',
                stat.trend === 'up' ? 'text-green-500' : 'text-red-500',
              )}
            >
              {stat.change}
            </span>
          </div>
        </div>
        <div
          className={cn(
            'w-12 h-12 rounded-lg flex items-center justify-center',
            'bg-[var(--primary)]/10',
            'group-hover:bg-[var(--primary)]/20 transition-colors',
          )}
        >
          <stat.icon className="w-6 h-6 text-[var(--primary)]" />
        </div>
      </div>
    </motion.div>
  );
};

// Tab Button Component
const TabButton = ({ tab, isActive, onClick }) => {
  return (
    <button
      onClick={onClick}
      className={cn(
        'relative px-6 py-3 font-medium text-sm transition-all duration-200',
        'hover:text-[var(--primary)]',
        isActive ? 'text-[var(--primary)]' : 'text-[var(--text-secondary)]',
      )}
    >
      <span className="relative z-10 flex items-center gap-2">
        <tab.icon className="w-4 h-4" />
        {tab.label}
      </span>
      {isActive && (
        <motion.div
          layoutId="activeTab"
          className="absolute inset-0 bg-[var(--primary)]/10 rounded-lg"
          transition={{ type: 'spring', duration: 0.5 }}
        />
      )}
      {isActive && (
        <motion.div
          layoutId="activeTabBorder"
          className="absolute bottom-0 left-0 right-0 h-0.5 bg-[var(--primary)]"
          transition={{ type: 'spring', duration: 0.5 }}
        />
      )}
    </button>
  );
};

// Resource Card Component
const ResourceCard = ({ resource, delay }) => {
  const [isExpanded, setIsExpanded] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay, duration: 0.3 }}
      whileHover={{ y: -4 }}
      className={cn(
        'relative rounded-xl overflow-hidden',
        'bg-[var(--card)] border border-[var(--border)]/50',
        'hover:shadow-xl transition-all duration-300',
        'group',
      )}
    >
      <div className="p-6 cursor-pointer" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="flex items-start justify-between mb-4">
          <div
            className={cn(
              'w-12 h-12 rounded-lg flex items-center justify-center',
              'bg-gradient-to-br from-[var(--primary)] to-[var(--primary)]/80',
              'shadow-lg group-hover:shadow-[var(--primary)]/30',
            )}
          >
            <resource.icon className="w-6 h-6 text-white" />
          </div>
          <motion.div animate={{ rotate: isExpanded ? 90 : 0 }} transition={{ duration: 0.2 }}>
            <ChevronRight className="w-5 h-5 text-[var(--text-secondary)]" />
          </motion.div>
        </div>

        <h3 className="text-lg font-semibold mb-2 text-[var(--text)]">{resource.title}</h3>
        <p className="text-sm text-[var(--text-secondary)] line-clamp-2">{resource.description}</p>

        {resource.badge && (
          <div className="mt-3">
            <span
              className={cn(
                'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium',
                'bg-[var(--primary)]/10 text-[var(--primary)]',
              )}
            >
              {resource.badge}
            </span>
          </div>
        )}
      </div>

      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="overflow-hidden"
          >
            <div className="px-6 pb-6 border-t border-[var(--border)]/30 pt-4">
              <div className="space-y-2">
                {resource.links.map((link, idx) => (
                  <a
                    key={idx}
                    href={link.url}
                    target={link.external ? '_blank' : undefined}
                    rel={link.external ? 'noopener noreferrer' : undefined}
                    className={cn(
                      'flex items-center justify-between p-3 rounded-lg',
                      'hover:bg-[var(--background-secondary)] transition-colors',
                      'text-sm text-[var(--text)]',
                    )}
                  >
                    <span>{link.name}</span>
                    {link.external ? (
                      <ExternalLink className="w-4 h-4 text-[var(--text-secondary)]" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-[var(--text-secondary)]" />
                    )}
                  </a>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

// Activity Timeline Component
const ActivityTimeline = ({ activities }) => {
  return (
    <div className="space-y-4">
      {activities.map((activity, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="flex gap-4"
        >
          <div className="relative">
            <div
              className={cn(
                'w-8 h-8 rounded-full flex items-center justify-center',
                'bg-[var(--primary)]/10',
              )}
            >
              <activity.icon className="w-4 h-4 text-[var(--primary)]" />
            </div>
            {index < activities.length - 1 && (
              <div className="absolute top-8 left-1/2 -translate-x-1/2 w-0.5 h-12 bg-[var(--border)]" />
            )}
          </div>
          <div className="flex-1 pb-8">
            <p className="text-sm font-medium text-[var(--text)]">{activity.title}</p>
            <p className="text-xs text-[var(--text-secondary)] mt-1">{activity.time}</p>
          </div>
        </motion.div>
      ))}
    </div>
  );
};

// Command Palette Component
const CommandPalette = ({ isOpen, onClose, items }) => {
  const [search, setSearch] = useState('');
  const inputRef = useRef(null);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  const filteredItems = items.filter(
    (item) =>
      item.title.toLowerCase().includes(search.toLowerCase()) ||
      item.description.toLowerCase().includes(search.toLowerCase()),
  );

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/60 z-50"
            onClick={onClose}
          />
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: -20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: -20 }}
            className="fixed top-20 left-1/2 -translate-x-1/2 w-full max-w-2xl z-50"
          >
            <div
              className={cn(
                'bg-[var(--card)] rounded-xl',
                'border border-[var(--border)]',
                'shadow-2xl overflow-hidden',
              )}
            >
              <div className="p-4 border-b border-[var(--border)]">
                <div className="flex items-center gap-3">
                  <Search className="w-5 h-5 text-[var(--text-secondary)]" />
                  <input
                    ref={inputRef}
                    type="text"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    placeholder="Search admin tools..."
                    className={cn(
                      'flex-1 bg-transparent outline-none',
                      'text-[var(--text)] placeholder-[var(--text-secondary)]',
                    )}
                  />
                  <kbd className="px-2 py-1 text-xs rounded bg-[var(--background-secondary)] text-[var(--text-secondary)]">
                    ESC
                  </kbd>
                </div>
              </div>

              <div className="max-h-96 overflow-y-auto p-2">
                {filteredItems.length === 0 ? (
                  <div className="p-8 text-center text-[var(--text-secondary)]">
                    No results found
                  </div>
                ) : (
                  <div className="space-y-1">
                    {filteredItems.map((item, index) => (
                      <motion.a
                        key={index}
                        href={item.url}
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: index * 0.05 }}
                        className={cn(
                          'block p-3 rounded-lg',
                          'hover:bg-[var(--background-secondary)]',
                          'transition-colors cursor-pointer',
                        )}
                        onClick={onClose}
                      >
                        <div className="flex items-center gap-3">
                          <item.icon className="w-5 h-5 text-[var(--primary)]" />
                          <div className="flex-1">
                            <p className="text-sm font-medium text-[var(--text)]">{item.title}</p>
                            <p className="text-xs text-[var(--text-secondary)]">
                              {item.description}
                            </p>
                          </div>
                        </div>
                      </motion.a>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default function TabBasedAdminView({ adminTools = [] }) {
  const [activeTab, setActiveTab] = useState('sops');
  const [showCommandPalette, setShowCommandPalette] = useState(false);

  // Stats data
  const stats = [
    {
      label: 'Active SOPs',
      value: '24',
      change: '+12%',
      trend: 'up',
      icon: FileText,
      trendIcon: TrendingUp,
    },
    {
      label: 'Team Members',
      value: '156',
      change: '+8%',
      trend: 'up',
      icon: Users,
      trendIcon: TrendingUp,
    },
    {
      label: 'Forms Processed',
      value: '1,234',
      change: '+23%',
      trend: 'up',
      icon: Files,
      trendIcon: TrendingUp,
    },
    {
      label: 'Avg. Processing Time',
      value: '2.4h',
      change: '-15%',
      trend: 'down',
      icon: Clock,
      trendIcon: TrendingUp,
    },
  ];

  // Recent activities
  const recentActivities = [
    {
      icon: CheckCircle,
      title: 'SOP 001 updated',
      time: '2 hours ago',
    },
    {
      icon: Users,
      title: 'New member onboarded',
      time: '5 hours ago',
    },
    {
      icon: Files,
      title: 'CF 52 form template revised',
      time: '1 day ago',
    },
    {
      icon: Wrench,
      title: 'System maintenance completed',
      time: '2 days ago',
    },
  ];

  // Tab configuration
  const tabs = [
    { id: 'sops', label: 'SOPs', icon: FileText },
    { id: 'onboarding', label: 'Onboarding', icon: GraduationCap },
    { id: 'forms', label: 'Forms', icon: FolderOpen },
    { id: 'resources', label: 'Resources', icon: Wrench },
  ];

  // Get current tab content
  const currentTabContent = adminTools.find((tool) => tool.id === activeTab);

  // All searchable items for command palette
  const allSearchableItems = adminTools.flatMap((tool) =>
    tool.links.map((link) => ({
      title: link.name,
      description: tool.title,
      url: link.url,
      icon: tool.icon,
    })),
  );

  // Keyboard shortcut for command palette
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setShowCommandPalette(true);
      }
      if (e.key === 'Escape') {
        setShowCommandPalette(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Header Section */}
      <div className="border-b border-[var(--border)]/50 bg-[var(--card)]/30 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-8">
            <div className="flex items-center justify-between mb-8">
              <div>
                <h1 className="text-3xl font-bold text-[var(--text)] flex items-center gap-3">
                  <Sparkles className="w-8 h-8 text-[var(--primary)]" />
                  Administrative Command Center
                </h1>
                <p className="text-[var(--text-secondary)] mt-2">
                  Centralized hub for all administrative operations
                </p>
              </div>

              <Button
                onClick={() => setShowCommandPalette(true)}
                variant="outline"
                className="hidden sm:flex items-center gap-2"
              >
                <Command className="w-4 h-4" />
                <span>Quick Search</span>
                <kbd className="ml-2 px-2 py-0.5 text-xs rounded bg-[var(--background-secondary)]">
                  ⌘K
                </kbd>
              </Button>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {stats.map((stat, index) => (
                <StatsCard key={index} stat={stat} delay={index * 0.1} />
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Left Column - Tabs and Content */}
          <div className="lg:col-span-3">
            {/* Tab Navigation */}
            <div className="flex space-x-1 mb-8 border-b border-[var(--border)]/50">
              {tabs.map((tab) => (
                <TabButton
                  key={tab.id}
                  tab={tab}
                  isActive={activeTab === tab.id}
                  onClick={() => setActiveTab(tab.id)}
                />
              ))}
            </div>

            {/* Tab Content */}
            <AnimatePresence mode="wait">
              {currentTabContent && (
                <motion.div
                  key={activeTab}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.3 }}
                >
                  <div className="mb-6">
                    <h2 className="text-2xl font-semibold text-[var(--text)]">
                      {currentTabContent.title}
                    </h2>
                    <p className="text-[var(--text-secondary)] mt-2">
                      {currentTabContent.description}
                    </p>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {currentTabContent.links.map((link, index) => (
                      <ResourceCard
                        key={index}
                        resource={{
                          ...currentTabContent,
                          title: link.name,
                          description: `Access ${link.name} documentation and resources`,
                          links: [link],
                          badge: link.external ? 'External' : 'Internal',
                        }}
                        delay={index * 0.1}
                      />
                    ))}
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Right Column - Activity Timeline */}
          <div className="lg:col-span-1">
            <div
              className={cn(
                'sticky top-8 rounded-xl',
                'bg-[var(--card)]/50 backdrop-blur-sm',
                'border border-[var(--border)]/50',
                'p-6',
              )}
            >
              <h3 className="text-lg font-semibold text-[var(--text)] mb-6 flex items-center gap-2">
                <Clock className="w-5 h-5 text-[var(--primary)]" />
                Recent Activity
              </h3>
              <ActivityTimeline activities={recentActivities} />
            </div>
          </div>
        </div>
      </div>

      {/* Command Palette */}
      <CommandPalette
        isOpen={showCommandPalette}
        onClose={() => setShowCommandPalette(false)}
        items={allSearchableItems}
      />
    </div>
  );
}
