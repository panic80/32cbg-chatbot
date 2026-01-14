import js from '@eslint/js';
import eslintConfigPrettier from 'eslint-config-prettier';
import globals from 'globals';
import reactPlugin from 'eslint-plugin-react';
import reactHooksPlugin from 'eslint-plugin-react-hooks';
import jsxA11yPlugin from 'eslint-plugin-jsx-a11y';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import tsParser from '@typescript-eslint/parser';

const reactRecommended = reactPlugin.configs.recommended ?? { rules: {} };
const reactJsxRuntime = reactPlugin.configs['jsx-runtime'] ?? { rules: {} };

export default [
  {
    ignores: [
      'dist',
      'node_modules',
      'rag-service/venv',
      'rag-service/venv/**',
      'rag-service/.venv',
      'rag-service/.venv/**',
      'rag-service/chroma_db',
      'rag-service/chroma_db/**',
      'rag-service/backups',
      'rag-service/backups/**',
      'rag-service/logs',
      'rag-service/logs/**',
      'rag-service/**/*.log',
      'rag-service/**/*.tar',
      'rag-service/**/*.tar.gz',
      'rag-service/**/*.zip',
      'rag-service/**/*.pyc',
      'rag-service/**/__pycache__',
      'source',
      'source/**',
      'src/components/ui/file-preview.tsx',
    ],
  },
  js.configs.recommended,
  {
    files: ['**/*.{js,jsx,ts,tsx,cjs,mjs}'],
    languageOptions: {
      ecmaVersion: 2023,
      sourceType: 'module',
      globals: {
        ...globals.browser,
        ...globals.node,
        ...globals.vitest,
      },
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    linterOptions: {
      reportUnusedDisableDirectives: 'off',
    },
    plugins: {
      react: reactPlugin,
      'react-hooks': reactHooksPlugin,
      'jsx-a11y': jsxA11yPlugin,
    },
    settings: {
      react: { version: 'detect' },
    },
    rules: {
      ...reactRecommended.rules,
      ...reactJsxRuntime.rules,
      'react/prop-types': 'off',
      'react/react-in-jsx-scope': 'off',
      'react/jsx-uses-react': 'off',
      'react/no-unescaped-entities': 'off',
      'react/display-name': 'off',
      'react/no-unknown-property': 'off',
      'react/jsx-no-undef': 'off',
      'react-hooks/rules-of-hooks': 'error',
      'react-hooks/exhaustive-deps': 'warn',
      'jsx-a11y/no-autofocus': 'off',
      'no-unused-vars': 'off',
      'no-empty': 'off',
      'no-undef': 'error',
      'no-redeclare': 'off',
      'no-useless-escape': 'off',
      'no-constant-binary-expression': 'off',
    },
  },
  {
    files: ['**/*.{ts,tsx}'],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        ecmaFeatures: { jsx: true },
      },
    },
    plugins: {
      '@typescript-eslint': tsPlugin,
    },
    rules: {
      '@typescript-eslint/no-unused-vars': 'off',
      '@typescript-eslint/no-explicit-any': 'warn',
      '@typescript-eslint/consistent-type-imports': 'off',
      'no-undef': 'off', // TypeScript handles this
    },
  },
  eslintConfigPrettier,
];
