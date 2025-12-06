import { render, screen } from '@testing-library/react';
import App from './App';
import { describe, it, expect } from 'vitest';
import '@testing-library/jest-dom'; // Ensure matchers are available

describe('App', () => {
    it('renders Vite + React text', () => {
        render(<App />);
        expect(screen.getByText('Vite + React')).toBeInTheDocument();
    });
});
