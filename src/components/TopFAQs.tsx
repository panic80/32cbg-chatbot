import React, { useEffect, useState } from 'react';
import { getTopQuestions } from '../api/questionAnalysis';

const TopFAQs = ({ visible }) => {
  const [faqs, setFaqs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [expandedIndex, setExpandedIndex] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(Date.now());

  // Create a custom event for FAQ updates
  useEffect(() => {
    const handleQuestionAdded = () => {
      setLastUpdate(Date.now());
    };

    window.addEventListener('questionAdded', handleQuestionAdded);
    return () => window.removeEventListener('questionAdded', handleQuestionAdded);
  }, []);

  // Fetch FAQs when visibility changes or when a new question is added
  useEffect(() => {
    let mounted = true;
    let pollInterval;

    const fetchFAQs = async () => {
      if (!mounted || !visible) return;

      try {
        setLoading(true);
        const questions = await getTopQuestions();
        if (mounted) {
          setFaqs(questions);
          setError(null);
        }
      } catch (err) {
        console.error('Error fetching FAQs:', err);
        if (mounted) {
          setError('Failed to load frequently asked questions');
        }
      } finally {
        if (mounted) {
          setLoading(false);
        }
      }
    };

    if (visible) {
      fetchFAQs();
      // Poll more frequently when visible
      pollInterval = setInterval(fetchFAQs, 5000);
    }

    return () => {
      mounted = false;
      if (pollInterval) {
        clearInterval(pollInterval);
      }
    };
  }, [visible, lastUpdate]);
  if (loading) {
    return (
      <div className="max-w-3xl mx-auto p-6">
        <div className="animate-pulse space-y-6">
          <div className="h-8 bg-gray-100 dark:bg-gray-700 rounded-md w-48"></div>
          {[...Array(3)].map((_, i) => (
            <div key={i} className="space-y-3">
              <div className="h-6 bg-gray-100 dark:bg-gray-700 rounded-md w-full"></div>
              <div className="h-4 bg-gray-50 dark:bg-gray-600 rounded-md w-2/3"></div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-3xl mx-auto p-6">
        <div className="bg-red-50 dark:bg-red-900/20 text-red-500 dark:text-red-400 rounded-lg p-4">
          {error}
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-4xl mx-auto px-4">
      <h2 className="text-3xl font-semibold text-gray-800 dark:text-white mb-6">
        Frequently Asked Questions
      </h2>
      {faqs.length === 0 ? (
        <div className="text-center py-12">
          <p className="text-gray-500 dark:text-gray-400 text-lg">
            No frequently asked questions yet.
          </p>
        </div>
      ) : (
        <div className="space-y-3">
          {faqs.map((faq, index) => (
            <div
              key={faq.id}
              className={`group bg-[#1e293b] rounded-lg shadow-sm hover:shadow-md transition-all duration-200 ${
                expandedIndex === index ? 'ring-2 ring-blue-500' : ''
              }`}
            >
              <div
                className="cursor-pointer p-5 hover:bg-[#334155] transition-colors duration-200 text-white"
                onClick={() => setExpandedIndex(expandedIndex === index ? null : index)}
              >
                <div className="flex items-start gap-4">
                  <div className="flex-shrink-0 w-8 h-8 rounded-md bg-blue-600 text-white font-medium flex items-center justify-center">
                    {index + 1}
                  </div>
                  <div className="flex-grow">
                    <p className="text-white text-lg font-medium">{faq.text}</p>
                    <div className="mt-2 flex items-center gap-2">
                      <div className="flex-grow h-1 bg-[#334155] rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-blue-500 to-blue-600 dark:from-blue-400 dark:to-blue-500 rounded-full transition-all duration-300"
                          style={{
                            width: `${Math.min((faq.count / Math.max(...faqs.map((f) => f.count))) * 100, 100)}%`,
                          }}
                        ></div>
                      </div>
                      <span className="text-sm text-gray-500 dark:text-gray-300 whitespace-nowrap flex items-center gap-1">
                        <svg
                          className="w-4 h-4"
                          fill="none"
                          stroke="currentColor"
                          viewBox="0 0 24 24"
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            strokeWidth={2}
                            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                          />
                        </svg>
                        {faq.count} {faq.count === 1 ? 'time' : 'times'}
                      </span>
                    </div>

                    {faq.variants && faq.variants.length > 0 && (
                      <div
                        className={`mt-3 overflow-hidden transition-all duration-300 ${
                          expandedIndex === index ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'
                        }`}
                      >
                        <p className="text-sm font-medium text-gray-500 dark:text-gray-300 mb-2">
                          Similar questions:
                        </p>
                        <ul className="space-y-2">
                          {faq.variants.map((variant, i) => (
                            <li
                              key={i}
                              className="text-sm text-gray-600 dark:text-gray-300 pl-3 border-l-2 border-blue-200 dark:border-[#1e293b] hover:border-blue-400 dark:hover:border-blue-400 transition-colors"
                            >
                              {variant}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                  {faq.variants && faq.variants.length > 0 && (
                    <button
                      className={`w-7 h-7 flex items-center justify-center rounded-md bg-gray-50 dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors duration-200 ${
                        expandedIndex === index ? 'rotate-180' : ''
                      }`}
                    >
                      <svg
                        className="w-4 h-4 text-gray-500 dark:text-gray-300"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M19 9l-7 7-7-7"
                        />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TopFAQs;
