# app/frontend/

React 18 + Vite 5 web application. Provides a mobile-style interface for live camera OCR, image upload, and voice feedback.

---

## Features

- **Live Camera tab** — streams webcam frames over WebSocket, plays detected text as audio in real time
- **Upload tab** — drag-and-drop or browse an image file, runs OCR via HTTP POST
- **About tab** — project description and tech stack
- **Stats bar** — running counts of spoken/silenced regions and rolling average latency
- **i18n** — English and French, persisted in `localStorage`
- **API health polling** — header indicator updates every 10 seconds

---

## Running locally

```bash
npm install
npm run dev        # development server at http://localhost:5173
npm run build      # production build → dist/
npm run preview    # preview production build locally
```

---

## Environment variables

Create `.env` in this directory:

```env
VITE_API_URL=http://localhost:8000
```

If `VITE_API_URL` is not set, the app defaults to the production HuggingFace Space:
`https://mocamaradata-visionlens-api.hf.space`

The WebSocket URL is derived automatically by replacing `http` with `ws`.

---

## Structure

```
src/
├── App.jsx               Root component — layout, language state, stats aggregation
├── App.css               Global styles
├── main.jsx              Vite entry point
├── components/           UI panels and tabs
├── hooks/                Camera, WebSocket, and audio state logic
├── i18n/                 English and French string bundles
└── constants/
    └── api.js            API endpoint URLs (derived from VITE_API_URL)
```

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `react` | ^18.2.0 | UI framework |
| `react-dom` | ^18.2.0 | DOM rendering |
| `vite` | ^5.2.0 | Build tool and dev server |
| `@vitejs/plugin-react` | ^4.2.1 | JSX transform |
