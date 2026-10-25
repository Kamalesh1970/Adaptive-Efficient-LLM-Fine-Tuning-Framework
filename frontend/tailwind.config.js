/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        studio: {
          bg: '#0a0d14',
          card: '#111622',
          cardHover: '#161d2d',
          cardBorder: '#1e2638',
          border: '#1e2638',
          borderSubtle: '#182030',
          input: '#0d121c',
          accent: '#10b981',
          accentHover: '#059669',
          teal: '#14b8a6',
          muted: '#94a3b8',
          text: '#f1f5f9',
          textSecondary: '#cbd5e1',
          sidebar: '#0d111a',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      }
    },
  },
  plugins: [],
}
