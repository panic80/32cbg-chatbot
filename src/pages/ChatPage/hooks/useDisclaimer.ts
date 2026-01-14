import { useEffect, useState } from 'react';
import { StorageKeys } from '@/constants/storage';
import { getLocalStorageItem, setLocalStorageItem } from '@/utils/storage';

export const useDisclaimer = () => {
  const [showDisclaimer, setShowDisclaimer] = useState(false);

  useEffect(() => {
    // Get or initialize visit count
    const visitCountStr = getLocalStorageItem(StorageKeys.visitCount);
    const visitCount = visitCountStr ? parseInt(visitCountStr, 10) : 0;
    const newVisitCount = visitCount + 1;

    // Store updated visit count
    setLocalStorageItem(StorageKeys.visitCount, newVisitCount.toString());

    // Show disclaimer every 3 visits
    if (newVisitCount % 3 === 1) {
      setShowDisclaimer(true);
    }
  }, []);

  return { showDisclaimer, setShowDisclaimer };
};
