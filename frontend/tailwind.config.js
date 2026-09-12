/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: ["Manrope", "ui-sans-serif", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "monospace"]
      },
      colors: {
        ink: "#101411",
        shell: "#f4f1ea",
        steel: "#25455b",
        moss: "#44624a",
        ember: "#c95f39",
        line: "#d9d3c5"
      },
      boxShadow: {
        panel: "0 18px 60px rgba(26, 31, 26, 0.12)"
      }
    }
  },
  plugins: []
};
