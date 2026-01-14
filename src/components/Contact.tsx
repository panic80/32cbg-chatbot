import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

const Contact: React.FC = () => {
  const navigate = useNavigate();

  useEffect(() => {
    window.location.href = 'mailto:g8@sent.com?subject=Contact from Unofficial G8 Webpage';
    navigate('/');
  }, [navigate]);

  return null;
};

export default Contact;
