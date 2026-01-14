import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import LandingPage from '..';
import { ThemeProvider } from '@/context/ThemeContext';

const renderLandingPage = () =>
  render(
    <ThemeProvider>
      <MemoryRouter>
        <LandingPage />
      </MemoryRouter>
    </ThemeProvider>,
  );

describe('LandingPage', () => {
  it('renders hero headline and quick ask chips', () => {
    renderLandingPage();

    expect(
      screen.getByRole('heading', { name: /32 CBG G8 Administration Hub/i }),
    ).toBeInTheDocument();

    expect(screen.getByRole('button', { name: /mileage rates/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /per diem rates/i })).toBeInTheDocument();
  });

  it('exposes key feature links', () => {
    renderLandingPage();

    expect(
      screen.getByRole('link', {
        name: /Policy Assistant - Interactive, RAG powered AI chat/i,
      }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole('link', {
        name: /OPI Contacts - Find FSC & FMC contact information/i,
      }),
    ).toBeInTheDocument();
  });
});
