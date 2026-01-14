import React, { useState, useRef } from 'react';
import LoadingScreen from '../components/LoadingScreen';

const LoadingDebugPage = () => {
  const [showLoader, setShowLoader] = useState(false);
  const loadTime = useRef(0);
  const [stats, setStats] = useState({});

  const toggleLoader = (duration) => {
    const start = performance.now();
    setShowLoader(true);

    setTimeout(() => {
      const end = performance.now();
      setShowLoader(false);
      setStats({
        duration: end - start,
        frameRate: (loadTime.current / (end - start)) * 1000,
      });
    }, duration);
  };

  return (
    <div className="debug-container">
      <div className="debug-controls">
        <button onClick={() => toggleLoader(1000)}>Test 1s Load</button>
        <button onClick={() => toggleLoader(3000)}>Test 3s Load</button>
        <button onClick={() => toggleLoader(5000)}>Test 5s Load</button>
      </div>

      {stats.duration && (
        <div className="debug-stats">
          <h3>Performance Metrics</h3>
          <p>Total duration: {stats.duration.toFixed(2)}ms</p>
          <p>Frame rate: {stats.frameRate.toFixed(2)}fps</p>
        </div>
      )}

      {showLoader && <LoadingScreen />}
    </div>
  );
};

export default LoadingDebugPage;
