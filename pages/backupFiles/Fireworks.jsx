import React from 'react';
import Confetti from 'react-confetti';

const Fireworks = ({ show }) => {
  if (!show) return null;
  return <Confetti numberOfPieces={100} />;
};

export default Fireworks;