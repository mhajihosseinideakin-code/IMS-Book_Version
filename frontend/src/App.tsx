import { Routes, Route } from "react-router-dom";
import FourState from "@/pages/FourState";

// One <Route> per page in src/pages; BrowserRouter already wraps this in main.tsx.
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<FourState />} />
    </Routes>
  );
}
