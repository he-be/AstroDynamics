import '@testing-library/jest-dom';

// ResizeObserver Polyfill for R3F
class ResizeObserver {
    observe() { }
    unobserve() { }
    disconnect() { }
}
window.ResizeObserver = ResizeObserver;
