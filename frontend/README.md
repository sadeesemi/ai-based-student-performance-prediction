# Module 03 dashboard (Create React App)

```bash
npm install
npm start        # http://localhost:3000  (or run ../run-dev.bat on Windows)
```

Routes: `/login`, `/profiling` (Module 01 output), `/prediction` (Module 02 output),
`/recommendations` (Module 03 - this module).

Data comes from `public/data`, which the Python pipeline writes. If those files are
missing, run the backend first:

```bash
cd ../backend/recommendation_module && python main.py
```

`src/services/api.js` is the only place that touches data. Set
`REACT_APP_USE_BACKEND=true` in a `.env` file to switch to the Flask API instead.
