import React from 'react';
import styled from 'styled-components';

const ImageContainer = styled.div`
  // Add your custom styles here
  
`;

const ImagePreview = ({ src }) => (
  <ImageContainer>{src && <img src={src} alt="Uploaded" />}Drop files here or click to upload</ImageContainer>
);

export default ImagePreview;