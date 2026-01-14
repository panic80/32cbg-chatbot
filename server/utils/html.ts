import * as cheerio from 'cheerio';
import { getLogger } from '../services/logger.js';

const logger = getLogger('utils:html');

/**
 * Process HTML content to extract main text while preserving structure hints.
 * Copied from server/main.js to ensure identical behavior and logs.
 */
export const processContent = (html: string): string => {
  try {
    logger.debug('Starting HTML processing with cheerio');

    const $ = cheerio.load(html, {
      xmlMode: false,
    });

    logger.debug('Cheerio loaded HTML successfully');

    const scriptCount = $('script').length;
    const styleCount = $('style').length;
    const headerCount = $('header').length;
    const footerCount = $('footer').length;
    const navCount = $('nav').length;

    logger.debug('Element counts before removal', {
      scriptCount,
      styleCount,
      headerCount,
      footerCount,
      navCount,
    });

    $('script, style, header, footer, nav').remove();

    let mainContent = '';
    const contentSelectors = [
      'main',
      'article',
      '.content',
      '#content',
      '.main-content',
      '.article-content',
      'div[role="main"]',
    ];

    for (const selector of contentSelectors) {
      if ($(selector).length > 0) {
        logger.debug('Found content using selector', { selector });
        mainContent = $(selector).text();
        break;
      }
    }

    if (!mainContent || mainContent.trim().length < 100) {
      logger.debug('Content selectors did not yield sufficient content, falling back to body');
      mainContent = $('body').text();
    }

    logger.debug('Raw extracted text length', { length: mainContent.length });

    const processedText = mainContent
      .replace(/\s+/g, ' ')
      .replace(/(\d+\.\d+\.?\d*)(\s+)/g, '\n$1$2')
      .replace(/(SECTION|Chapter|CHAPTER|Part|PART)\s+(\d+)/gi, '\n$1 $2')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(
        /([Ll]unch).+?(\d{1,2}[:\.]\d{2}).+?(\d{1,2}[:\.]\d{2})/g,
        (_match, meal, start, end) => {
          return `${meal} may be claimed when duty travel extends through the period of ${start} to ${end}`;
        },
      )
      .replace(/([.!?])\s+/g, '$1\n')
      .trim();

    logger.debug('Processed text length', { length: processedText.length });

    return processedText;
  } catch (error) {
    logger.error('Error processing HTML content', { error });
    try {
      return html
        .replace(/<[^>]*>/g, ' ')
        .replace(/\s+/g, ' ')
        .replace(/(\d+\.\d+\.?\d*)/g, '\n$1')
        .trim();
    } catch (fallbackError) {
      logger.error('Even fallback processing failed', { error: fallbackError });
      throw new Error('Content processing failed completely');
    }
  }
};
