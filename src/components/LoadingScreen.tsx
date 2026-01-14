import React, { useEffect, useState } from 'react';
import { useLoading, LoadingStage, getStagePercentage } from '../context/LoadingContext';
import { AnimatedButton } from './ui/animated-button';
import { Card } from './ui/card';
import { RefreshCw, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import './LoadingScreen.css';

interface StageIndicatorProps {
  currentStage: LoadingStage;
  stage: LoadingStage;
  label: string;
}

const StageIndicator: React.FC<StageIndicatorProps> = ({ currentStage, stage, label }) => {
  const getStatus = () => {
    const stages: LoadingStage[] = ['url-scanning', 'parsing', 'validation', 'complete'];
    const currentIndex = stages.indexOf(currentStage);
    const stageIndex = stages.indexOf(stage);

    if (currentStage === 'error') return 'error';
    if (stageIndex < currentIndex) return 'completed';
    if (stageIndex === currentIndex) return 'active';
    return 'pending';
  };

  const status = getStatus();
  const Icon =
    status === 'completed'
      ? CheckCircle
      : status === 'error'
        ? AlertCircle
        : status === 'active'
          ? Loader2
          : null;

  return (
    <div className="stage-indicator-container">
      <div className={`stage-indicator ${status}`}>
        {Icon && <Icon className={`stage-icon ${status === 'active' ? 'animate-spin' : ''}`} />}
      </div>
      <span className="stage-label">{label}</span>
    </div>
  );
};

const LoadingScreen: React.FC = () => {
  const { state, dispatch } = useLoading();
  const [showDetails, setShowDetails] = useState(false);

  // Auto-show details after 3 seconds if still loading
  useEffect(() => {
    if (state.isLoading && !state.error) {
      const timer = setTimeout(() => setShowDetails(true), 3000);
      return () => clearTimeout(timer);
    }
  }, [state.isLoading, state.error]);

  const handleRetry = () => {
    dispatch({ type: 'RESET' });
    setShowDetails(false);
    // Trigger retry event
    window.dispatchEvent(new CustomEvent('retryUrlParsing'));
  };

  if (!state.isLoading) return null;

  const stages = [
    { stage: 'url-scanning' as LoadingStage, label: 'Scanning URL' },
    { stage: 'parsing' as LoadingStage, label: 'Parsing Content' },
    { stage: 'validation' as LoadingStage, label: 'Validating Data' },
    { stage: 'complete' as LoadingStage, label: 'Complete' },
  ];

  return (
    <div className="loading-screen">
      {/* Animated background */}
      <div className="loading-background">
        <div className="loading-gradient-1" />
        <div className="loading-gradient-2" />
      </div>

      <Card className="loading-content glass animate-fade-up">
        <div className="loading-icon-container">
          <div className="loading-icon-wrapper">
            <Loader2 className="loading-main-icon animate-spin" />
          </div>
        </div>

        <h2 className="loading-title h3 text-2xl">Processing URL</h2>
        <p className="loading-message body-base">{state.message}</p>

        {/* Progress bar with shimmer effect */}
        <div className="loading-bar-container">
          <div className="loading-bar">
            <div className="loading-bar-fill" style={{ width: `${state.progress}%` }} />
            <div className="loading-bar-shimmer" />
          </div>
          <span className="loading-percentage">{Math.round(state.progress)}%</span>
        </div>

        {/* Stage indicators with transitions */}
        {(showDetails || state.error) && (
          <div className="loading-stages animate-fade-up">
            {stages.map(({ stage, label }) => (
              <StageIndicator key={stage} currentStage={state.stage} stage={stage} label={label} />
            ))}
          </div>
        )}

        {/* Error state with improved design */}
        {state.error && (
          <div className="loading-error-container animate-fade-up">
            <div className="loading-error-icon">
              <AlertCircle className="w-6 h-6" />
            </div>
            <p className="loading-error-message">{state.error}</p>
            <AnimatedButton
              onClick={handleRetry}
              variant="secondary"
              size="sm"
              className="mt-4 gap-2"
              ripple
            >
              <RefreshCw className="w-4 h-4" />
              Try Again
            </AnimatedButton>
          </div>
        )}

        {/* Success state animation */}
        {state.stage === 'complete' && !state.error && (
          <div className="loading-success animate-scale-in">
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        )}
      </Card>
    </div>
  );
};

export default LoadingScreen;
