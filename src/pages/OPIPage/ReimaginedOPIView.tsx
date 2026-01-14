import React, { useState, useMemo, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Mail,
  Building2,
  Crown,
  ChevronRight,
  Grid3x3,
  LayoutList,
  UserCircle2,
  Briefcase,
  MapPin,
  Filter,
  X,
  CheckCircle2,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Button } from '@/components/ui/button';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';

// Modern contact card with split design
const ModernContactCard = ({ contact, type, onClick, index }) => {
  const [isHovered, setIsHovered] = useState(false);

  const getBgGradient = () => {
    // Vibrant teal/cyan for FSC, Gold for FMC
    if (type === 'FSC') return 'from-teal-500/15 to-cyan-500/15';
    if (type === 'FMC') return 'from-amber-500/15 to-yellow-500/15';
    return 'from-primary/5 to-primary/10';
  };

  const getIconBg = () => {
    // Vibrant teal for FSC, Gold/amber for FMC
    if (type === 'FSC') return 'bg-gradient-to-br from-teal-600 to-cyan-600';
    if (type === 'FMC') return 'bg-gradient-to-br from-amber-500 to-yellow-600';
    return 'bg-[var(--primary)]';
  };

  const getBorderHover = () => {
    // Distinct hover colors with glow
    if (type === 'FSC') return 'hover:border-teal-500/60 hover:shadow-teal-500/20';
    if (type === 'FMC') return 'hover:border-amber-500/60 hover:shadow-amber-500/20';
    return 'hover:border-primary/50';
  };

  const getAccentColor = () => {
    // For text highlights on hover
    if (type === 'FSC') return 'group-hover:text-teal-600';
    if (type === 'FMC') return 'group-hover:text-amber-600';
    return 'group-hover:text-primary';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
      onClick={onClick}
      className="h-full"
    >
      <Card
        className={cn(
          'group relative overflow-hidden border border-[var(--border)] cursor-pointer',
          'transition-all duration-300 hover:shadow-xl hover:scale-[1.02]',
          'bg-gradient-to-br h-full backdrop-blur-sm',
          'glass',
          getBgGradient(),
          getBorderHover(),
        )}
      >
        <CardContent className="p-0 h-full flex flex-col">
          {/* Top section with colored accent bar (Teal=FSC, Gold=FMC) */}
          <div className={cn('h-1 w-full', getIconBg())} />

          <div className="p-4 flex flex-col flex-1">
            {/* Header with small type badge and leadership indicator */}
            <div className="flex items-start justify-between mb-3">
              <div
                className={cn('px-2 py-0.5 rounded-md text-xs font-bold text-white', getIconBg())}
              >
                {type || 'N/A'}
              </div>
              {contact.isLeadership && (
                <Crown className="w-4 h-4 text-[var(--primary)]" title="Leadership" />
              )}
            </div>

            {/* Contact name - BIGGER */}
            <h3
              className={cn(
                'text-2xl font-bold text-foreground mb-1 transition-colors leading-tight',
                getAccentColor(),
              )}
            >
              {contact.name}
            </h3>

            {/* Role - smaller */}
            <p className="text-xs text-muted-foreground mb-3">{contact.role}</p>

            {/* Units - show all with emphasis */}
            {contact.units && contact.units.length > 0 && (
              <div className="mb-3">
                <div className="flex flex-wrap gap-1.5">
                  {contact.units.map((unit, idx) => (
                    <Badge
                      key={idx}
                      className={cn(
                        'text-xs font-semibold px-2 py-1 shadow-sm',
                        type === 'FSC'
                          ? 'bg-teal-100 text-teal-800 border-teal-300 dark:bg-teal-900/30 dark:text-teal-200 dark:border-teal-700'
                          : type === 'FMC'
                            ? 'bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-900/30 dark:text-amber-200 dark:border-amber-700'
                            : 'bg-primary/10 text-primary border-primary/30',
                      )}
                    >
                      {unit}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Email - compact at bottom */}
            <motion.div
              animate={{ x: isHovered ? 3 : 0 }}
              className={cn(
                'flex items-center gap-1.5 text-xs font-medium mt-auto pt-2 border-t border-[var(--border)] transition-colors',
                type === 'FSC'
                  ? 'text-teal-600 hover:text-teal-700'
                  : type === 'FMC'
                    ? 'text-amber-600 hover:text-amber-700'
                    : 'text-primary hover:text-primary/80',
              )}
            >
              <Mail className="w-3.5 h-3.5" />
              <span className="truncate text-[11px]">{contact.email}</span>
              <ChevronRight className="w-3 h-3 ml-auto flex-shrink-0" />
            </motion.div>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
};

// Compact list view for contacts
const CompactContactRow = ({ contact, index }) => {
  const isFSC = contact.role && contact.role.includes('FSC');
  const isFMC = contact.role && contact.role.includes('FMC');

  const getBorderColor = () => {
    if (isFSC) return 'border-l-teal-600 hover:border-l-teal-500';
    if (isFMC) return 'border-l-amber-500 hover:border-l-amber-400';
    return 'border-l-primary hover:border-l-primary/80';
  };

  const getTextColor = () => {
    if (isFSC) return 'text-teal-600';
    if (isFMC) return 'text-amber-600';
    return 'text-primary';
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.02 }}
      className={cn(
        'group p-3 rounded-r-lg border-l-4 border-y border-r border-[var(--border)]',
        'hover:shadow-md glass hover:scale-[1.005] transition-all duration-200',
        getBorderColor(),
      )}
    >
      <div className="flex items-center justify-between gap-3">
        <div className="flex items-center gap-3 flex-1 min-w-0">
          {/* Compact type badge */}
          <div className={cn('px-2 py-0.5 rounded text-[10px] font-bold', getTextColor())}>
            {isFSC ? 'FSC' : isFMC ? 'FMC' : 'N/A'}
          </div>

          {/* Info */}
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <h4 className="font-bold text-lg text-foreground truncate">{contact.name}</h4>
              {contact.isLeadership && (
                <Crown className="w-3.5 h-3.5 text-[var(--primary)] flex-shrink-0" />
              )}
            </div>
            <p className="text-xs text-muted-foreground mb-2">{contact.role}</p>
            {contact.units && contact.units.length > 0 && (
              <div className="flex flex-wrap gap-1.5 mt-1.5">
                {contact.units.map((unit, idx) => (
                  <Badge
                    key={idx}
                    className={cn(
                      'text-xs font-semibold px-2 py-0.5 shadow-sm',
                      isFSC
                        ? 'bg-teal-100 text-teal-800 border-teal-300 dark:bg-teal-900/30 dark:text-teal-200 dark:border-teal-700'
                        : isFMC
                          ? 'bg-amber-100 text-amber-800 border-amber-300 dark:bg-amber-900/30 dark:text-amber-200 dark:border-amber-700'
                          : 'bg-primary/10 text-primary border-primary/30',
                    )}
                  >
                    {unit}
                  </Badge>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Email */}
        <a
          href={`mailto:${contact.email}`}
          className={cn(
            'hidden lg:flex items-center gap-1.5 text-xs font-medium group-hover:translate-x-1 transition-transform',
            getTextColor(),
          )}
          onClick={(e) => e.stopPropagation()}
        >
          <Mail className="w-3.5 h-3.5" />
          <span className="max-w-[200px] truncate">{contact.email}</span>
          <ChevronRight className="w-3 h-3" />
        </a>

        {/* Mobile email button */}
        <a
          href={`mailto:${contact.email}`}
          className={cn(
            'lg:hidden p-1.5 rounded-lg hover:bg-primary/10 transition-colors',
            getTextColor(),
          )}
          onClick={(e) => e.stopPropagation()}
        >
          <Mail className="w-4 h-4" />
        </a>
      </div>
    </motion.div>
  );
};

// Main reimagined view component
export default function ReimaginedOPIView({
  unitContacts = {},
  fscContacts = [],
  fmcContacts = [],
  contactView: initialView = 'all',
  selectedUnit = '',
  searchTerm = '',
  setSelectedUnit = () => {},
  setSearchTerm = () => {},
  setContactView = () => {},
}) {
  const [localView, setLocalView] = useState(initialView);
  const [localSearchTerm, setLocalSearchTerm] = useState(searchTerm);
  const [viewStyle, setViewStyle] = useState('card'); // 'card' is default
  const [filterType, setFilterType] = useState('all'); // 'all', 'leadership', 'sections'

  // Combine all contacts
  const allContacts = useMemo(() => [...fscContacts, ...fmcContacts], [fscContacts, fmcContacts]);

  // Filter units
  const allUnits = useMemo(() => Object.keys(unitContacts).sort(), [unitContacts]);
  const filteredUnits = useMemo(() => {
    const term = localSearchTerm.toLowerCase();
    return allUnits.filter((unit) => unit.toLowerCase().includes(term));
  }, [allUnits, localSearchTerm]);

  // Get contacts based on view and filter
  const getDisplayContacts = useCallback(() => {
    let contacts = [];

    switch (localView) {
      case 'fsc':
        contacts = fscContacts;
        break;
      case 'fmc':
        contacts = fmcContacts;
        break;
      case 'all':
      default:
        contacts = allContacts;
        break;
    }

    // Apply filters
    if (filterType === 'leadership') {
      contacts = contacts.filter((c) => c.isLeadership);
    } else if (filterType === 'sections') {
      contacts = contacts.filter((c) => !c.isLeadership);
    }

    return contacts;
  }, [localView, fscContacts, fmcContacts, allContacts, filterType]);

  const displayContacts = useMemo(() => getDisplayContacts(), [getDisplayContacts]);

  const handleContactClick = useCallback((email) => {
    window.location.href = `mailto:${email}`;
  }, []);

  // Quick stats
  const stats = useMemo(
    () => ({
      total: allContacts.length,
      fsc: fscContacts.length,
      fmc: fmcContacts.length,
      units: allUnits.length,
      leadership: allContacts.filter((c) => c.isLeadership).length,
    }),
    [allContacts, fscContacts, fmcContacts, allUnits],
  );

  return (
    <div className="space-y-6">
      {/* Unified Navigation Bar */}
      <div className="flex items-center justify-between gap-4 glass p-2 rounded-lg border border-[var(--border)]">
        {/* Main navigation tabs */}
        <div className="flex flex-wrap gap-1 flex-1">
          <Button
            variant={localView === 'all' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setLocalView('all')}
            className="text-xs"
          >
            All Contacts
          </Button>
          <Button
            variant={localView === 'fsc' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setLocalView('fsc')}
            className="text-xs"
          >
            <span className="w-2 h-2 rounded-full bg-gradient-to-br from-teal-600 to-cyan-600 mr-1.5" />
            FSC
          </Button>
          <Button
            variant={localView === 'fmc' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setLocalView('fmc')}
            className="text-xs"
          >
            <span className="w-2 h-2 rounded-full bg-gradient-to-br from-amber-500 to-yellow-600 mr-1.5" />
            FMC
          </Button>
          <Button
            variant={localView === 'search' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setLocalView('search')}
            className="text-xs"
          >
            <Search className="w-3 h-3 mr-1" />
            By Unit
          </Button>
        </div>

        {/* View style toggle */}
        <div className="flex gap-1 border-l border-[var(--border)] pl-2">
          <Button
            variant={viewStyle === 'card' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setViewStyle('card')}
            className="w-8 h-8 p-0"
            title="Card View"
          >
            <Grid3x3 className="w-3.5 h-3.5" />
          </Button>
          <Button
            variant={viewStyle === 'list' ? 'default' : 'ghost'}
            size="sm"
            onClick={() => setViewStyle('list')}
            className="w-8 h-8 p-0"
            title="List View"
          >
            <LayoutList className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      {/* Search Unit View */}
      {localView === 'search' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          <Card className="glass border border-[var(--border)] shadow-md">
            <CardContent className="p-6 space-y-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-muted-foreground" />
                <Input
                  type="text"
                  value={localSearchTerm}
                  onChange={(e) => setLocalSearchTerm(e.target.value)}
                  placeholder="Search for a unit..."
                  className="pl-10 h-12 text-base"
                />
                {localSearchTerm && (
                  <button
                    onClick={() => setLocalSearchTerm('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
                  >
                    <X className="w-5 h-5" />
                  </button>
                )}
              </div>

              <Select value={selectedUnit} onValueChange={setSelectedUnit}>
                <SelectTrigger className="h-12 text-base">
                  <SelectValue placeholder="Select a unit from the list" />
                </SelectTrigger>
                <SelectContent>
                  {filteredUnits.map((unit) => (
                    <SelectItem key={unit} value={unit} className="text-base">
                      <div className="flex items-center gap-2">
                        <MapPin className="w-4 h-4" />
                        {unit}
                      </div>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>

              {filteredUnits.length === 0 && localSearchTerm && (
                <p className="text-center text-sm text-muted-foreground py-4">
                  No units found matching "{localSearchTerm}"
                </p>
              )}
            </CardContent>
          </Card>

          {/* Selected unit results */}
          {selectedUnit && unitContacts[selectedUnit] && (
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
              <div className="mb-4 flex items-center gap-2 text-sm text-muted-foreground">
                <CheckCircle2 className="w-4 h-4 text-green-500" />
                Showing contacts for <strong className="text-foreground">{selectedUnit}</strong>
              </div>

              <div className="grid gap-4 md:grid-cols-2 auto-rows-fr">
                <ModernContactCard
                  contact={{
                    name: unitContacts[selectedUnit].fsc,
                    role: 'Financial Services Cell (FSC)',
                    email: unitContacts[selectedUnit].fscEmail,
                    units: [selectedUnit],
                  }}
                  type="FSC"
                  onClick={() => handleContactClick(unitContacts[selectedUnit].fscEmail)}
                  index={0}
                />
                <ModernContactCard
                  contact={{
                    name: unitContacts[selectedUnit].fmc,
                    role: 'Financial Management Cell (FMC)',
                    email: unitContacts[selectedUnit].fmcEmail,
                    units: [selectedUnit],
                  }}
                  type="FMC"
                  onClick={() => handleContactClick(unitContacts[selectedUnit].fmcEmail)}
                  index={1}
                />
              </div>
            </motion.div>
          )}
        </motion.div>
      )}

      {/* Contacts display (all, fsc, fmc views) */}
      {localView !== 'search' && (
        <div className="space-y-6">
          {/* Contacts grid/list */}
          <AnimatePresence mode="wait">
            {viewStyle === 'card' ? (
              <motion.div
                key="card-view"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="grid gap-4 md:grid-cols-2 lg:grid-cols-3 auto-rows-fr"
              >
                {displayContacts.map((contact, index) => {
                  const isFSC = contact.role && contact.role.includes('FSC');
                  const isFMC = contact.role && contact.role.includes('FMC');
                  const type = isFSC ? 'FSC' : isFMC ? 'FMC' : null;

                  return (
                    <ModernContactCard
                      key={index}
                      contact={contact}
                      type={type}
                      onClick={() => handleContactClick(contact.email)}
                      index={index}
                    />
                  );
                })}
              </motion.div>
            ) : (
              <motion.div
                key="list-view"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="space-y-2"
              >
                {displayContacts.map((contact, index) => (
                  <CompactContactRow key={index} contact={contact} index={index} />
                ))}
              </motion.div>
            )}
          </AnimatePresence>

          {/* Empty state */}
          {displayContacts.length === 0 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-12"
            >
              <UserCircle2 className="w-16 h-16 mx-auto text-muted-foreground mb-4" />
              <p className="text-lg text-muted-foreground">No contacts found</p>
            </motion.div>
          )}
        </div>
      )}
    </div>
  );
}
