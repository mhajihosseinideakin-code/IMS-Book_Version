import { useEffect } from "react";
import { Routes, Route } from "react-router-dom";
import FourState from "@/pages/FourState";

// The existing IMS Platform (Flask explorer.html, served by the backend) is the
// primary application. The public root redirects into it. The verified
// Four-State Stabilising-MRC workflow is a React analysis surface embedded
// natively inside the platform's Explorer (see explorer.html -> showStabilizingMRC).
function PlatformRedirect() {
  useEffect(() => {
    window.location.replace("/explorer.html");
  }, []);
  return null;
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<PlatformRedirect />} />
      <Route path="/mrc" element={<FourState />} />
    </Routes>
  );
}
