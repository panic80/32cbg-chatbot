import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  BookOpen,
  CircleHelp,
  ClipboardList,
  Lightbulb,
  Search,
  Filter,
  Bookmark,
  Clock,
  ChevronRight,
  ExternalLink,
  Tag,
  FileText,
  GraduationCap,
  FolderOpen,
  Wrench,
  LayoutGrid,
  List,
  Download,
  Share,
  Star,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Button } from '@/components/ui/button';

// Search Bar Component
const SearchBar = ({ onSearch, placeholder = 'Search library...' }) => {
  const [searchTerm, setSearchTerm] = useState('');

  const handleSearch = (e) => {
    const value = e.target.value;
    setSearchTerm(value);
    onSearch(value);
  };

  return (
    <div className={cn('relative w-full max-w-2xl mx-auto')}>
      <div
        className={cn(
          'relative flex items-center',
          'bg-[var(--card)] rounded-xl',
          'border border-[var(--border)]/50',
          'shadow-sm hover:shadow-md transition-shadow',
          'group',
        )}
      >
        <Search className="absolute left-4 w-5 h-5 text-[var(--text-secondary)]" />
        <input
          type="text"
          value={searchTerm}
          onChange={handleSearch}
          placeholder={placeholder}
          className={cn(
            'w-full pl-12 pr-4 py-4',
            'bg-transparent outline-none',
            'text-[var(--text)] placeholder-[var(--text-secondary)]',
            'text-lg',
          )}
        />
        <div className="absolute right-4 flex items-center gap-2">
          <kbd className="hidden sm:inline-flex px-2 py-1 text-xs rounded bg-[var(--background-secondary)] text-[var(--text-secondary)]">
            ⌘K
          </kbd>
        </div>
      </div>
    </div>
  );
};

// Category Filter Component
const CategoryFilter = ({ categories, activeCategory, onCategoryChange }) => {
  return (
    <div className="flex flex-wrap gap-2">
      <Button
        variant={activeCategory === 'all' ? 'default' : 'outline'}
        size="sm"
        onClick={() => onCategoryChange('all')}
        className="gap-2"
      >
        <LayoutGrid className="w-4 h-4" />
        All Resources
      </Button>
      {categories.map((category) => (
        <Button
          key={category.id}
          variant={activeCategory === category.id ? 'default' : 'outline'}
          size="sm"
          onClick={() => onCategoryChange(category.id)}
          className="gap-2"
        >
          <category.icon className="w-4 h-4" />
          {category.label}
        </Button>
      ))}
    </div>
  );
};

// Resource Card Component - Library Style
const LibraryResourceCard = ({ resource, viewMode, delay }) => {
  const [isBookmarked, setIsBookmarked] = useState(false);
  const [rating, setRating] = useState(0);

  if (viewMode === 'list') {
    return (
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay }}
        className={cn(
          'group flex items-center gap-4 p-4 rounded-lg',
          'bg-[var(--card)] border border-[var(--border)]/30',
          'hover:shadow-md transition-all duration-200',
          'cursor-pointer',
        )}
      >
        <div
          className={cn(
            'w-12 h-12 rounded-lg flex items-center justify-center flex-shrink-0',
            'bg-gradient-to-br from-[var(--primary)]/20 to-[var(--primary)]/10',
            'group-hover:from-[var(--primary)]/30 group-hover:to-[var(--primary)]/20',
            'transition-colors',
          )}
        >
          <resource.icon className="w-6 h-6 text-[var(--primary)]" />
        </div>

        <div className="flex-1 min-w-0">
          <h3 className="font-semibold text-[var(--text)] truncate">{resource.title}</h3>
          <p className="text-sm text-[var(--text-secondary)] truncate">{resource.description}</p>
        </div>

        <div className="flex items-center gap-3">
          <span className="text-xs text-[var(--text-secondary)]">
            {resource.readTime || '5 min read'}
          </span>
          <ChevronRight className="w-5 h-5 text-[var(--text-secondary)] group-hover:text-[var(--primary)] transition-colors" />
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay }}
      whileHover={{ y: -4 }}
      className={cn(
        'group relative h-full',
        'bg-[var(--card)] rounded-xl',
        'border border-[var(--border)]/30',
        'hover:shadow-xl transition-all duration-300',
        'overflow-hidden',
      )}
    >
      {/* Card Header with Icon */}
      <div
        className={cn(
          'relative h-32 bg-gradient-to-br',
          'from-[var(--primary)]/10 to-[var(--primary)]/5',
          'border-b border-[var(--border)]/30',
        )}
      >
        <div className="absolute inset-0 flex items-center justify-center">
          <resource.icon className="w-16 h-16 text-[var(--primary)] opacity-20" />
        </div>

        {/* Quick Actions */}
        <div className="absolute top-4 right-4 flex gap-2">
          <button
            onClick={(e) => {
              e.stopPropagation();
              setIsBookmarked(!isBookmarked);
            }}
            className={cn(
              'p-2 rounded-lg',
              'bg-white/80 dark:bg-black/80 backdrop-blur-sm',
              'hover:bg-white dark:hover:bg-black',
              'transition-all duration-200',
            )}
          >
            {isBookmarked ? (
              <Bookmark className="w-4 h-4 text-[var(--primary)] fill-current" />
            ) : (
              <Bookmark className="w-4 h-4 text-[var(--text-secondary)]" />
            )}
          </button>
        </div>

        {/* Category Badge */}
        <div className="absolute bottom-4 left-4">
          <span
            className={cn(
              'inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full',
              'bg-white/80 dark:bg-black/80 backdrop-blur-sm',
              'text-xs font-medium text-[var(--primary)]',
            )}
          >
            <Tag className="w-3 h-3" />
            {resource.category || 'General'}
          </span>
        </div>
      </div>

      {/* Card Content */}
      <div className="p-6">
        <h3 className="text-lg font-semibold text-[var(--text)] mb-2 line-clamp-2">
          {resource.title}
        </h3>
        <p className="text-sm text-[var(--text-secondary)] mb-4 line-clamp-3">
          {resource.description}
        </p>

        {/* Meta Information */}
        <div className="flex items-center justify-between text-xs text-[var(--text-secondary)] mb-4">
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <Clock className="w-3 h-3" />
              {resource.readTime || '5 min'}
            </span>
            <span className="flex items-center gap-1">
              <FileText className="w-3 h-3" />
              {resource.pageCount || '10'} pages
            </span>
          </div>
        </div>

        {/* Rating */}
        <div className="flex items-center gap-1 mb-4">
          {[1, 2, 3, 4, 5].map((star) => (
            <button
              key={star}
              onClick={(e) => {
                e.stopPropagation();
                setRating(star);
              }}
              className="transition-transform hover:scale-110"
            >
              {star <= rating ? (
                <Star className="w-4 h-4 text-yellow-400 fill-yellow-400" />
              ) : (
                <Star className="w-4 h-4 text-[var(--text-secondary)]" />
              )}
            </button>
          ))}
          <span className="text-xs text-[var(--text-secondary)] ml-2">
            {rating > 0 ? `${rating}.0` : 'Rate this'}
          </span>
        </div>

        {/* Action Buttons */}
        <div className="flex gap-2">
          <Button
            variant="default"
            size="sm"
            className="flex-1 gap-2"
            onClick={() =>
              window.open(
                resource.primaryLink?.url || '#',
                resource.primaryLink?.external ? '_blank' : '_self',
              )
            }
          >
            <BookOpen className="w-4 h-4" />
            View Resource
          </Button>
          <Button variant="outline" size="sm" className="gap-2">
            <Share className="w-4 h-4" />
          </Button>
        </div>
      </div>
    </motion.div>
  );
};

// Quick Access Section
const QuickAccessSection = ({ items }) => {
  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {items.map((item, index) => (
        <motion.a
          key={index}
          href={item.url}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          className={cn(
            'p-4 rounded-xl text-center',
            'bg-[var(--card)] border border-[var(--border)]/30',
            'hover:shadow-md hover:border-[var(--primary)]/30',
            'transition-all duration-200 group',
          )}
        >
          <div
            className={cn(
              'w-12 h-12 mx-auto mb-3 rounded-lg',
              'bg-gradient-to-br from-[var(--primary)]/20 to-[var(--primary)]/10',
              'flex items-center justify-center',
              'group-hover:scale-110 transition-transform',
            )}
          >
            <item.icon className="w-6 h-6 text-[var(--primary)]" />
          </div>
          <h4 className="font-medium text-sm text-[var(--text)]">{item.label}</h4>
          <p className="text-xs text-[var(--text-secondary)] mt-1">{item.count} items</p>
        </motion.a>
      ))}
    </div>
  );
};

// Recently Viewed Section
const RecentlyViewedSection = ({ items }) => {
  return (
    <div className="space-y-3">
      {items.map((item, index) => (
        <motion.a
          key={index}
          href={item.url}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className={cn(
            'flex items-center gap-3 p-3 rounded-lg',
            'hover:bg-[var(--background-secondary)]',
            'transition-colors group',
          )}
        >
          <div className="w-8 h-8 rounded bg-[var(--primary)]/10 flex items-center justify-center flex-shrink-0">
            <item.icon className="w-4 h-4 text-[var(--primary)]" />
          </div>
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-[var(--text)] truncate">{item.title}</p>
            <p className="text-xs text-[var(--text-secondary)]">{item.time}</p>
          </div>
          <ChevronRight className="w-4 h-4 text-[var(--text-secondary)] group-hover:text-[var(--primary)] transition-colors" />
        </motion.a>
      ))}
    </div>
  );
};

export default function DigitalLibraryView({ adminTools = [] }) {
  const [searchTerm, setSearchTerm] = useState('');
  const [activeCategory, setActiveCategory] = useState('all');
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'list'

  // Transform adminTools to library resources
  const libraryResources = adminTools.map((tool) => ({
    ...tool,
    category:
      tool.id === 'sops'
        ? 'Procedures'
        : tool.id === 'onboarding'
          ? 'Training'
          : tool.id === 'forms'
            ? 'Templates'
            : 'References',
    readTime:
      tool.id === 'sops'
        ? '15 min'
        : tool.id === 'onboarding'
          ? '30 min'
          : tool.id === 'forms'
            ? '5 min'
            : '10 min',
    pageCount:
      tool.id === 'sops'
        ? '25'
        : tool.id === 'onboarding'
          ? '40'
          : tool.id === 'forms'
            ? '8'
            : '15',
    primaryLink: tool.links[0],
  }));

  // Categories for filtering
  const categories = [
    { id: 'procedures', label: 'Procedures', icon: ClipboardList },
    { id: 'training', label: 'Training', icon: GraduationCap },
    { id: 'templates', label: 'Templates', icon: FileText },
    { id: 'references', label: 'References', icon: BookOpen },
  ];

  // Quick access items
  const quickAccessItems = [
    { label: 'SOPs', count: 24, icon: ClipboardList, url: '#' },
    { label: 'How-To Guides', count: 18, icon: Lightbulb, url: '#' },
    { label: 'FAQs', count: 42, icon: CircleHelp, url: '#' },
    { label: 'Templates', count: 15, icon: FileText, url: '#' },
  ];

  // Recently viewed items
  const recentlyViewed = [
    { title: 'Leave Request Process', time: '2 hours ago', icon: FileText, url: '#' },
    { title: 'Travel Claim Guide', time: '1 day ago', icon: ClipboardList, url: '#' },
    { title: 'New Member Checklist', time: '3 days ago', icon: GraduationCap, url: '#' },
  ];

  // Filter resources based on search and category
  const filteredResources = libraryResources.filter((resource) => {
    const matchesSearch =
      resource.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      resource.description.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesCategory =
      activeCategory === 'all' || resource.category.toLowerCase() === activeCategory;
    return matchesSearch && matchesCategory;
  });

  return (
    <div className="min-h-screen bg-[var(--background)]">
      {/* Hero Section */}
      <div className="relative bg-gradient-to-b from-[var(--primary)]/5 to-transparent">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-center mb-8"
          >
            <div className="flex items-center justify-center gap-3 mb-4">
              <BookOpen className="w-12 h-12 text-[var(--primary)]" />
              <h1 className="text-4xl font-bold text-[var(--text)]">Resource Library</h1>
            </div>
            <p className="text-lg text-[var(--text-secondary)] max-w-2xl mx-auto">
              Your comprehensive digital library for procedures, guides, and administrative
              resources
            </p>
          </motion.div>

          {/* Search Bar */}
          <SearchBar onSearch={setSearchTerm} placeholder="Search for SOPs, guides, templates..." />
        </div>
      </div>

      {/* Main Content */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          {/* Main Content Area */}
          <div className="lg:col-span-3">
            {/* Filters and View Options */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
              <CategoryFilter
                categories={categories}
                activeCategory={activeCategory}
                onCategoryChange={setActiveCategory}
              />

              <div className="flex items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setViewMode('grid')}
                  className={cn(viewMode === 'grid' && 'bg-[var(--primary)] text-white')}
                >
                  <LayoutGrid className="w-4 h-4" />
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setViewMode('list')}
                  className={cn(viewMode === 'list' && 'bg-[var(--primary)] text-white')}
                >
                  <List className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Results Count */}
            <p className="text-sm text-[var(--text-secondary)] mb-6">
              Showing {filteredResources.length} resources
            </p>

            {/* Resources Grid/List */}
            <div
              className={cn(
                viewMode === 'grid' ? 'grid grid-cols-1 md:grid-cols-2 gap-6' : 'space-y-3',
              )}
            >
              {filteredResources.map((resource, index) => (
                <LibraryResourceCard
                  key={resource.id}
                  resource={resource}
                  viewMode={viewMode}
                  delay={index * 0.1}
                />
              ))}
            </div>

            {filteredResources.length === 0 && (
              <div className="text-center py-12">
                <BookOpen className="w-12 h-12 mx-auto text-[var(--text-secondary)] mb-4" />
                <p className="text-[var(--text-secondary)]">
                  No resources found matching your criteria
                </p>
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            {/* Quick Access */}
            <div
              className={cn('rounded-xl p-6', 'bg-[var(--card)] border border-[var(--border)]/30')}
            >
              <h3 className="font-semibold text-[var(--text)] mb-4 flex items-center gap-2">
                <Lightbulb className="w-5 h-5 text-[var(--primary)]" />
                Quick Access
              </h3>
              <QuickAccessSection items={quickAccessItems} />
            </div>

            {/* Recently Viewed */}
            <div
              className={cn('rounded-xl p-6', 'bg-[var(--card)] border border-[var(--border)]/30')}
            >
              <h3 className="font-semibold text-[var(--text)] mb-4 flex items-center gap-2">
                <Clock className="w-5 h-5 text-[var(--primary)]" />
                Recently Viewed
              </h3>
              <RecentlyViewedSection items={recentlyViewed} />
            </div>

            {/* Help Section */}
            <div
              className={cn(
                'rounded-xl p-6',
                'bg-gradient-to-br from-[var(--primary)]/10 to-[var(--primary)]/5',
                'border border-[var(--primary)]/20',
              )}
            >
              <h3 className="font-semibold text-[var(--text)] mb-2">Need Help?</h3>
              <p className="text-sm text-[var(--text-secondary)] mb-4">
                Can't find what you're looking for? Contact our support team.
              </p>
              <Button variant="outline" size="sm" className="w-full gap-2">
                <CircleHelp className="w-4 h-4" />
                Get Support
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
