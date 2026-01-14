// Initialize IndexedDB database with enhanced error handling and logging
export const initDB = (
  dbName: string,
  version: number,
  upgradeCallback: (db: IDBDatabase) => void,
): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    console.log(`Initializing IndexedDB: ${dbName} (v${version})`);
    const request = indexedDB.open(dbName, version);

    request.onerror = (event) => {
      const error = (event.target as IDBOpenDBRequest).error;
      console.error('IndexedDB error:', error);
      reject(error);
    };

    request.onsuccess = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      console.log(`Successfully opened IndexedDB: ${dbName}`);

      // Add error handler for all database operations
      db.onerror = (event) => {
        console.error('Database error:', (event.target as IDBDatabase).error);
      };

      resolve(db);
    };

    request.onupgradeneeded = (event) => {
      console.log(`Upgrading IndexedDB: ${dbName}`);
      const db = (event.target as IDBOpenDBRequest).result;
      try {
        upgradeCallback(db);
        console.log('Database upgrade completed successfully');
      } catch (error) {
        console.error('Error during database upgrade:', error);
        throw error;
      }
    };

    request.onblocked = () => {
      console.warn('Database upgrade blocked. Please close other tabs and refresh.');
    };
  });
};

// Generic function to add data to any store with enhanced error handling
export const addToStore = async <T>(
  db: IDBDatabase,
  storeName: string,
  data: T,
): Promise<IDBValidKey> => {
  return new Promise((resolve, reject) => {
    console.log(`Adding data to store: ${storeName}`, data);

    const transaction = db.transaction(storeName, 'readwrite');
    const store = transaction.objectStore(storeName);

    transaction.oncomplete = () => {
      console.log(`Successfully added data to ${storeName}`);
    };

    transaction.onerror = (event) => {
      const error = (event.target as IDBTransaction).error;
      console.error(`Error in transaction for ${storeName}:`, error);
      reject(error);
    };

    try {
      const request = store.add(data);

      request.onsuccess = (event) => {
        const key = (event.target as IDBRequest).result;
        console.log(`Data added successfully with key: ${key}`);
        resolve(key);
      };

      request.onerror = (event) => {
        const error = (event.target as IDBRequest).error;
        console.error(`Error adding data to ${storeName}:`, error);
        reject(error);
      };
    } catch (error) {
      console.error(`Exception while adding data to ${storeName}:`, error);
      reject(error);
    }
  });
};

// Generic function to get all data from any store with enhanced error handling
export const getAllFromStore = async <T>(db: IDBDatabase, storeName: string): Promise<T[]> => {
  return new Promise((resolve, reject) => {
    console.log(`Getting all data from store: ${storeName}`);

    const transaction = db.transaction(storeName, 'readonly');
    const store = transaction.objectStore(storeName);

    transaction.oncomplete = () => {
      console.log(`Successfully completed read transaction for ${storeName}`);
    };

    transaction.onerror = (event) => {
      const error = (event.target as IDBTransaction).error;
      console.error(`Error in read transaction for ${storeName}:`, error);
      reject(error);
    };

    try {
      const request = store.getAll();

      request.onsuccess = (event) => {
        const results = (event.target as IDBRequest<T[]>).result;
        console.log(`Retrieved ${results.length} records from ${storeName}`);
        resolve(results);
      };

      request.onerror = (event) => {
        const error = (event.target as IDBRequest).error;
        console.error(`Error getting data from ${storeName}:`, error);
        reject(error);
      };
    } catch (error) {
      console.error(`Exception while getting data from ${storeName}:`, error);
      reject(error);
    }
  });
};

// Generic function to update data in any store
export const updateInStore = async <T>(
  db: IDBDatabase,
  storeName: string,
  data: T,
): Promise<IDBValidKey> => {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.put(data);

    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

// Generic function to delete data from any store
export const deleteFromStore = async (
  db: IDBDatabase,
  storeName: string,
  key: IDBValidKey,
): Promise<void> => {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.delete(key);

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};

// Generic function to clear all data from any store
export const clearStore = async (db: IDBDatabase, storeName: string): Promise<void> => {
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(storeName, 'readwrite');
    const store = transaction.objectStore(storeName);
    const request = store.clear();

    request.onsuccess = () => resolve();
    request.onerror = () => reject(request.error);
  });
};
