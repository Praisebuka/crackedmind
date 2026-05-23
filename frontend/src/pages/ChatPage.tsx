import React, { useState, useRef } from 'react';
import { Typography, Paper, Box, TextField, IconButton, Button, InputAdornment } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AttachFileIcon from '@mui/icons-material/AttachFile';
import axios from 'axios';

interface Message {
  sender: 'user' | 'agent';
  text: string;
}

const ChatPage: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSend = async () => {
    if (!input.trim()) return;
    setMessages([...messages, { sender: 'user', text: input }]);
    setInput('');
    // API call
    try {
      const res = await axios.post('http://localhost:8000/v1/task-a/simulate-review', { message: input });
      setMessages(msgs => [...msgs, { sender: 'agent', text: res.data.response }]);
      console.log(res);
    } catch {
      setMessages(msgs => [...msgs, { sender: 'agent', text: 'Error: Could not reach backend.' }]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        width: '100vw',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: 'linear-gradient(135deg, #1976d2 0%, #9c27b0 100%)',
        overflow: 'hidden',
        px: { xs: 1, sm: 2, md: 0 },
      }}
    >
      <Paper
        elevation={8}
        sx={{
          p: { xs: 1, sm: 2, md: 4 },
          borderRadius: 6,
          minWidth: { xs: '100%', sm: 420 },
          maxWidth: { xs: '100%', sm: 540 },
          minHeight: { xs: '90vh', sm: 540 },
          background: 'rgba(255,255,255,0.13)',
          backdropFilter: 'blur(16px)',
          border: '1.5px solid rgba(255,255,255,0.18)',
          display: 'flex',
          flexDirection: 'column',
        }}
      >
        <Typography
          variant="h4"
          gutterBottom
          align="center"
          sx={{
            fontWeight: 700,
            color: '#fff',
            textShadow: '0 2px 16px rgba(25, 118, 210, 0.18)',
            mb: 2,
          }}
        >
          Chat with CrackedMind Agent
        </Typography>
        <Box
          sx={{
            flex: 1,
            overflowY: 'auto',
            mb: 2,
            pr: 1,
            display: 'flex',
            flexDirection: 'column',
            gap: 1.5,
          }}
        >
          {messages.map((msg, idx) => (
            <Box
              key={idx}
              sx={{
                display: 'flex',
                justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              }}
            >
              <Box
                sx={{
                  px: 2.2,
                  py: 1.2,
                  borderRadius: 4,
                  maxWidth: '80%',
                  background:
                    msg.sender === 'user'
                      ? 'linear-gradient(90deg, #1976d2 60%, #9c27b0 100%)'
                      : 'rgba(255,255,255,0.85)',
                  color: msg.sender === 'user' ? '#fff' : '#222',
                  fontWeight: 500,
                  fontSize: 17,
                  boxShadow:
                    msg.sender === 'user'
                      ? '0 2px 12px 0 rgba(25, 118, 210, 0.12)'
                      : '0 2px 12px 0 rgba(156, 39, 176, 0.10)',
                  borderBottomRightRadius: msg.sender === 'user' ? 0 : 16,
                  borderBottomLeftRadius: msg.sender === 'user' ? 16 : 0,
                }}
              >
                {msg.text}
              </Box>
            </Box>
          ))}
        </Box>
        <Box style={{ display: 'flex', background: '#d2d2d2' }} alignItems="center" gap={1}>
          <input
            type="file"
            style={{ display: 'none' }}
            ref={fileInputRef}
            onChange={handleFileChange}
          />
          <IconButton
            color="primary"
            sx={{ background: 'rgba(25,118,210,0.08)' }}
            onClick={() => fileInputRef.current?.click()}
          >
            <AttachFileIcon />
          </IconButton>
          <TextField
            fullWidth
            variant="outlined"
            placeholder="Type your message..."
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            sx={{
              background: 'rgba(255,255,255,0.85)',
              borderRadius: 2,
            }}
            rows="4" aria-colspan="50"
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    color="primary"
                    sx={{ background: 'linear-gradient(90deg, #1976d2 60%, #9c27b0 100%)', color: '#fff', ml: 1, '&:hover': { background: 'linear-gradient(90deg, #1565c0 60%, #7b1fa2 100%)' } }}
                    onClick={handleSend}
                  >
                    <SendIcon />
                  </IconButton>
                </InputAdornment>
              ),
            }}
          />
          {file && <Button size="small" onClick={() => setFile(null)} sx={{ ml: 1 }}>Remove File</Button>}
        </Box>
        {file && (
          <Box mt={1}>
            <Typography variant="caption" color="#fff">Selected file: {file.name}</Typography>
          </Box>
        )}
      </Paper>
    </Box>
  );
};

export default ChatPage;
