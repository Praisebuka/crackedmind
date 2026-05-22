import React from 'react';
import { Container, Typography, Button, Box } from '@mui/material';
import { useNavigate } from 'react-router-dom';

const LandingPage: React.FC = () => {
  const navigate = useNavigate();
  return (
    <Container maxWidth="md" sx={{ mt: 10, textAlign: 'center' }}>
      <Typography variant="h2" gutterBottom>
        CrackedMind
      </Typography>
      <Typography variant="h5" color="text.secondary" gutterBottom>
        Next-generation User Modeling & Recommendation Platform
      </Typography>
      <Typography variant="body1" sx={{ mb: 4 }}>
        Simulate authentic reviews, deliver personalized recommendations, and experience secure, culturally-aware AI—all in one place.
      </Typography>
      <Box>
        <Button variant="contained" color="primary" size="large" onClick={() => navigate('/chat')}>
          Try the Agent
        </Button>
      </Box>
    </Container>
  );
};

export default LandingPage;
