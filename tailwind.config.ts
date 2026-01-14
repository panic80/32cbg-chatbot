import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  darkMode: ['class', 'class'],
  theme: {
    extend: {
      animation: {
        'fade-in': 'fadeIn 0.6s ease-out forwards',
        'fade-up': 'fadeUp 0.8s ease-out forwards',
        scale: 'scale 0.6s ease-out forwards',
        'float-up': 'floatUp 0.5s ease-out forwards',
        bounce: 'bounce 1s infinite',
        'typing-dot-bounce': 'typing-dot-bounce 1.25s ease-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': {
            opacity: '0',
          },
          '100%': {
            opacity: '1',
          },
        },
        fadeUp: {
          '0%': {
            opacity: '0',
            transform: 'translateY(20px)',
          },
          '100%': {
            opacity: '1',
            transform: 'translateY(0)',
          },
        },
        scale: {
          '0%': {
            opacity: '0',
            transform: 'scale(0.9)',
          },
          '100%': {
            opacity: '1',
            transform: 'scale(1)',
          },
        },
        floatUp: {
          '0%': {
            opacity: '0',
            transform: 'translate(-50%, -45%)',
          },
          '100%': {
            opacity: '1',
            transform: 'translate(-50%, -50%)',
          },
        },
        'typing-dot-bounce': {
          '0%,40%': {
            transform: 'translateY(0)',
          },
          '20%': {
            transform: 'translateY(-0.25rem)',
          },
        },
      },
      scale: {
        115: '1.15',
      },
      colors: {
        border: 'var(--border)',
        input: 'var(--input)',
        ring: 'var(--ring)',
        background: 'var(--background)',
        foreground: 'var(--foreground)',
        primary: {
          DEFAULT: 'var(--primary)',
          foreground: 'var(--primary-foreground)',
          hover: 'var(--primary-hover)',
          active: 'var(--primary-active)',
        },
        secondary: {
          DEFAULT: 'var(--secondary, var(--background-secondary))',
          foreground: 'var(--secondary-foreground, var(--foreground))',
        },
        destructive: {
          DEFAULT: 'var(--error)',
          foreground: '#ffffff',
          hover: 'var(--error-hover)',
        },
        muted: {
          DEFAULT: 'var(--muted)',
          foreground: 'var(--muted-foreground)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          foreground: 'var(--accent-foreground)',
        },
        card: {
          DEFAULT: 'var(--card)',
          foreground: 'var(--card-foreground)',
        },
        success: {
          DEFAULT: 'var(--success)',
          foreground: '#ffffff',
          hover: 'var(--success-hover)',
        },
        warning: {
          DEFAULT: 'var(--warning)',
          foreground: '#000000',
          hover: 'var(--warning-hover)',
        },
        error: {
          DEFAULT: 'var(--error)',
          foreground: '#ffffff',
          hover: 'var(--error-hover)',
        },
        info: {
          DEFAULT: 'var(--info)',
          foreground: '#ffffff',
          hover: 'var(--info-hover)',
        },
        shiki: {
          light: 'var(--shiki-light)',
          'light-bg': 'var(--shiki-light-bg)',
          dark: 'var(--shiki-dark)',
          'dark-bg': 'var(--shiki-dark-bg)',
        },
      },
      borderRadius: {
        lg: 'var(--radius, 0.5rem)',
        md: 'calc(var(--radius, 0.5rem) - 2px)',
        sm: 'calc(var(--radius, 0.5rem) - 4px)',
      },
    },
  },
  plugins: [],
} satisfies Config;
