import React from 'react';
import { Container, Typography, Button, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';


const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #1976d2 0%, #9c27b0 100%)',
        boxSizing: 'border-box',
        // Remove overflow and px to prevent horizontal scroll
      }}
    >
      <Box
        sx={{
          p: { xs: 2, sm: 3, md: 6 },
          borderRadius: 6,
          boxShadow: 8,
          background: 'rgba(255,255,255,0.13)',
          backdropFilter: 'blur(16px)',
          border: '1.5px solid rgba(255,255,255,0.18)',
          maxWidth: { xs: '100%', sm: 420, md: 520 },
          width: '100%',
          textAlign: 'center',
          boxSizing: 'border-box',
          minWidth: '700px'
        }}
      >
        <Typography
          variant="h2"
          sx={{
            fontWeight: 800,
            letterSpacing: '-2px',
            color: '#fff',
            mb: 2,
            textShadow: '0 4px 32px rgba(25, 118, 210, 0.25)',
          }}
        >
          CrackedMind
        </Typography>
        <Typography
          variant="h5"
          sx={{
            color: 'rgba(255,255,255,0.92)',
            mb: 2,
            fontWeight: 500,
          }}
        >
          Next-generation User Modeling & Recommendation Platform
        </Typography>
        <Typography
          variant="body1"
          sx={{
            color: 'rgba(255,255,255,0.85)',
            mb: 4,
            fontSize: 20,
            fontWeight: 400,
          }}
        >
          Simulate authentic reviews, deliver personalized recommendations, and experience secure, culturally-aware AI—all in one place.
        </Typography>
        <Button
          variant="contained"
          color="primary"
          size="large"
          sx={{
            px: 6,
            py: 1.5,
            fontWeight: 700,
            fontSize: 18,
            borderRadius: 3,
            boxShadow: '0 2px 16px 0 rgba(25, 118, 210, 0.18)',
            background: 'linear-gradient(90deg, #1976d2 60%, #9c27b0 100%)',
            transition: 'transform 0.2s',
            '&:hover': {
              transform: 'scale(1.04)',
              background: 'linear-gradient(90deg, #1565c0 60%, #7b1fa2 100%)',
            },
          }}
          onClick={() => navigate('/chat')}
        >
          Try our Agent
        </Button>
      </Box>
    </Box>
  );
};

export default LandingPage;
