import React, { useState, useRef } from 'react';
import { Container, Typography, Paper, Box, TextField, IconButton, Button, InputAdornment } from '@mui/material';
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
      const res = await axios.post('/api/agent', { message: input });
      setMessages(msgs => [...msgs, { sender: 'agent', text: res.data.response }]);
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
    <Container maxWidth="sm" sx={{ mt: 6 }}>
      <Typography variant="h4" gutterBottom align="center">
        Chat with CrackedMind Agent
      </Typography>
      <Paper sx={{ p: 2, minHeight: 400, mb: 2 }}>
        {messages.map((msg, idx) => (
          <Box key={idx} sx={{ textAlign: msg.sender === 'user' ? 'right' : 'left', mb: 1 }}>
            <Typography variant="body2" color={msg.sender === 'user' ? 'primary' : 'secondary'}>
              {msg.text}
            </Typography>
          </Box>
        ))}
      </Paper>
      <Box display="flex" alignItems="center" gap={1}>
        <input
          type="file"
          style={{ display: 'none' }}
          ref={fileInputRef}
          onChange={handleFileChange}
        />
        <IconButton color="primary" onClick={() => fileInputRef.current?.click()}>
          <AttachFileIcon />
        </IconButton>
        <TextField
          fullWidth
          variant="outlined"
          placeholder="Type your message..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && handleSend()}
          InputProps={{
            endAdornment: (
              <InputAdornment position="end">
                <IconButton color="primary" onClick={handleSend}>
                  <SendIcon />
                </IconButton>
              </InputAdornment>
            ),
          }}
        />
        {file && <Button size="small" onClick={() => setFile(null)}>Remove File</Button>}
      </Box>
      {file && (
        <Box mt={1}>
          <Typography variant="caption">Selected file: {file.name}</Typography>
        </Box>
      )}
    </Container>
  );
};

export default ChatPage;
