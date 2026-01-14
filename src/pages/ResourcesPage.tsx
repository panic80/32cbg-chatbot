import { TooltipProvider } from '@/components/ui/tooltip';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { EnhancedBackButton } from '@/components/ui/enhanced-back-button';
import { BackgroundEffects } from './ChatPage/components/BackgroundEffects';
import { useTheme } from '@/context/ThemeContext';
import LogoImage from '@/components/LogoImage';
import { FileText, ShieldCheck, FolderKanban, Lock, ArrowRight, Sun, Moon } from 'lucide-react';

const ResourcesPage: React.FC = () => {
  const { theme, toggleTheme } = useTheme();

  const resourceSections = [
    {
      title: 'SOP Library',
      description: 'Central index for approved SOPs and reference guides.',
      icon: FileText,
      items: ['Claims processing SOP', 'Travel entitlements SOP', 'Reserve pay SOP'],
    },
    {
      title: 'Templates & Forms',
      description: 'Downloadable templates with the latest version history.',
      icon: FolderKanban,
      items: ['Travel claim package', 'Mileage verification form', 'Advance request form'],
    },
    {
      title: 'Governance & Controls',
      description: 'Policy directives, risk controls, and compliance checkpoints.',
      icon: ShieldCheck,
      items: ['Audit checklist', 'Delegation of authorities', 'Control framework briefs'],
    },
  ];

  return (
    <TooltipProvider>
      <div className="relative min-h-screen overflow-x-hidden bg-[var(--background)] text-[var(--text)]">
        <BackgroundEffects />

        <div className="relative z-10 flex min-h-screen flex-col">
          <header className="sticky top-0 z-40 border-b border-[var(--border)] glass backdrop-blur-xl shadow-sm">
            <div className="flex h-16 items-center justify-between px-3 sm:px-6">
              <div className="flex items-center gap-3 sm:gap-4">
                <EnhancedBackButton to="/" label="Back" variant="minimal" size="sm" />
                <div className="h-8 w-px bg-border/50" />
                <div className="h-8 sm:h-9 md:h-10">
                  <LogoImage fitParent className="h-full w-auto" />
                </div>
                <div className="flex flex-col">
                  <span className="text-base font-bold text-foreground sm:text-xl md:text-2xl">
                    Resources Hub
                  </span>
                  <span className="hidden text-xs text-muted-foreground sm:block">
                    Standard Operating Procedures
                  </span>
                </div>
              </div>
              <div className="flex items-center gap-3">
                <Button
                  variant="outline"
                  size="icon"
                  onClick={toggleTheme}
                  className="h-11 w-11 rounded-lg border-2 shadow-md transition-all duration-200 hover:border-[var(--accent-foreground)] hover:bg-[var(--accent)] hover:shadow-lg"
                  aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                  {theme === 'dark' ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
                </Button>
              </div>
            </div>
          </header>

          <main className="flex-1">
            <div className="mx-auto flex w-full max-w-6xl flex-col gap-6 px-4 pb-10 pt-6 sm:px-6 sm:pt-10 lg:px-8 xl:px-10">
              <section className="rounded-3xl border border-[var(--border)]/60 bg-[var(--background)]/75 px-6 py-8 shadow-2xl backdrop-blur">
                <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
                  <div>
                    <h2 className="text-2xl font-semibold text-[var(--text)] sm:text-3xl">
                      Authenticated Resource Workspace
                    </h2>
                    <p className="mt-3 max-w-2xl text-sm text-[var(--text-secondary)] sm:text-base">
                      You are viewing the secured SOP catalogue. Content mirrors the OPI and Chat
                      layouts, optimized for administrators maintaining live documentation.
                    </p>
                  </div>
                  <div className="flex items-center gap-3 rounded-2xl border border-[var(--border)]/60 px-4 py-3 text-sm text-[var(--primary)]">
                    <Lock className="h-4 w-4" />
                    <div>
                      <p className="font-semibold uppercase tracking-wide">Access Control</p>
                      <p className="text-xs text-[var(--text-secondary)]">
                        Protected via Config Panel credentials
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              <section className="grid gap-6 lg:grid-cols-3">
                {resourceSections.map((section) => (
                  <Card
                    key={section.title}
                    className="border border-[var(--border)]/60 bg-[var(--background)]/80 backdrop-blur"
                  >
                    <CardHeader className="flex flex-row items-start gap-4">
                      <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-[var(--primary)]/15 text-[var(--primary)]">
                        <section.icon className="h-6 w-6" />
                      </div>
                      <div>
                        <CardTitle className="text-lg font-semibold text-[var(--text)]">
                          {section.title}
                        </CardTitle>
                        <CardDescription className="text-sm text-[var(--text-secondary)]">
                          {section.description}
                        </CardDescription>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-2 text-sm text-[var(--text-secondary)]">
                      {section.items.map((item) => (
                        <div
                          key={item}
                          className="flex items-center gap-2 rounded-lg border border-dashed border-[var(--border)]/60 px-3 py-2"
                        >
                          <ArrowRight className="h-3.5 w-3.5 text-[var(--primary)]" />
                          <span>{item}</span>
                        </div>
                      ))}
                    </CardContent>
                  </Card>
                ))}
              </section>

              <section className="grid gap-6 lg:grid-cols-[2fr,1fr]">
                <Card className="border border-[var(--border)]/60 bg-[var(--background)]/80 backdrop-blur">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg sm:text-xl">
                      <FileText className="h-5 w-5 text-[var(--primary)]" />
                      Publishing Checklist
                    </CardTitle>
                    <CardDescription>
                      Track the readiness workflow before releasing updates to the G8 community.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-[var(--text-secondary)]">
                    <p>Every SOP update should cover the following gates:</p>
                    <ul className="space-y-2">
                      <li className="flex items-start gap-2">
                        <span className="mt-1 h-2 w-2 rounded-full bg-[var(--primary)]" />
                        Validate latest policy references and authorities.
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="mt-1 h-2 w-2 rounded-full bg-[var(--primary)]" />
                        Ensure supporting templates are version-controlled.
                      </li>
                      <li className="flex items-start gap-2">
                        <span className="mt-1 h-2 w-2 rounded-full bg-[var(--primary)]" />
                        Coordinate comms with the OPI contact roster.
                      </li>
                    </ul>
                  </CardContent>
                </Card>

                <Card className="border border-[var(--border)]/60 bg-[var(--background)]/80 backdrop-blur">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-lg sm:text-xl">
                      <ShieldCheck className="h-5 w-5 text-[var(--primary)]" />
                      Compliance Snapshot
                    </CardTitle>
                    <CardDescription>
                      Overview of upcoming reviews and required confirmations.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3 text-sm text-[var(--text-secondary)]">
                    <div className="rounded-xl border border-[var(--border)]/60 px-4 py-3">
                      <p className="text-sm font-semibold text-[var(--text)]">
                        Q2 Evidence Collection
                      </p>
                      <p className="text-xs text-[var(--text-secondary)]">
                        Target completion: 15 May · Owner: FSC Compliance Cell
                      </p>
                    </div>
                    <div className="rounded-xl border border-[var(--border)]/60 px-4 py-3">
                      <p className="text-sm font-semibold text-[var(--text)]">Audit Prep</p>
                      <p className="text-xs text-[var(--text-secondary)]">
                        Pending notes from Open Findings tracker
                      </p>
                    </div>
                    <p className="text-xs text-[var(--text-secondary)]">
                      Need to raise an exception? Contact the Admin Tools team for expedited review.
                    </p>
                  </CardContent>
                </Card>
              </section>
            </div>
          </main>
        </div>
      </div>
    </TooltipProvider>
  );
};

export default ResourcesPage;
