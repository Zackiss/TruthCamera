import React from "react";
import styles from '../styles/Home.module.css';
import ReactMarkdown from 'react-markdown'

const Contact = () => {
  return (
    <div className={styles.container}>
      <main>
        <div className={styles.container2}></div>
        <h1 className={styles.title2}>
          This is <a href="/">TruthCamera</a>
        </h1>

        <p className={styles.description}>
          <code>A WEB3 solution for photo validation</code>
        </p>

        <div className={styles.grid}>
          <a href="/" className={styles.card}>
            <h3>Introduction</h3>
          </a>

        </div>

          {/* <div className={styles.card}>
            <DropzoneContainer {...getRootProps()}>
              <input {...getInputProps()} />
              <UploadMessage>Drop files here or click to upload</UploadMessage>
            </DropzoneContainer>
            
           <Fireworks show={showFireworks} />

            <ul>
              {uploadedFiles.map((file, index) => (
                <li key={index}>{file.name}</li>
              ))}
            </ul>
          </div>

        </div>

        <div>
          <Button onClick={handleUpload}>Verify</Button>
        </div> */}
        <div className={styles.container3}>
<div className={styles.card2}>
            The Truth Camera is a blockchain-based camera that provides trustful forensic evidence collection, self-certification of media public trust, and anti-AI forgery capabilities. It is designed to be used in situations where the authenticity and integrity of visual media are critical, such as in legal or journalistic contexts.
            </div>

<div className={styles.card2}>
            <h2 className={styles.mdh2}> 🔧 How it works</h2>

            The Trustful Camera uses blockchain technology to create a tamper-proof chain of custody for visual media. Each time a photo or video is taken with the camera, it is automatically timestamped and stored on a blockchain. This ensures that the media can be traced back to its source and that it has not been altered or manipulated in any way.

            In addition to the blockchain-based chain of custody, the Trustful Camera also includes anti-AI forgery capabilities. This involves embedding digital watermarks into the media that can be used to detect any attempts to alter or manipulate the media using AI or other digital tools.
            </div>

<div className={styles.card2}>
            <h2 className={styles.mdh2}>🗝 Features</h2>

            - Blockchain-based chain of custody
            - Anti-AI forgery capabilities
            - Tamper-proof timestamping
            - Secure storage on a decentralized blockchain network
            - Easy-to-use interface
            </div>

<div className={styles.card2}>
            <h2 className={styles.mdh2}>🛒 Getting started</h2> 

            To get started with the Trustful Camera, you will need to:

            1. Install the Trustful Camera app on your mobile IOS system device or Trustful Camera devices
            2. Start taking photos and videos with automatically stamped and anti-faked security measures

            Once you have taken photos or videos with the Trustful Camera, you can use the app to view and manage your media, including verifying the chain of custody and detecting any attempts at forgery.
            </div>

<div className={styles.card2}>
            <h2 className={styles.mdh2}> 📦 Contributing</h2>

            We welcome contributions to the Trustful Camera project. If you would like to contribute, please fork the repository and submit a pull request. You can also report issues or suggest new features using the GitHub issue tracker.
            </div>

<div className={styles.card2}>
            <h2 className={styles.mdh2}>🧾 License</h2>

            The Trustful Camera is released under the MIT license. See `LICENSE` for more information.
            The Trustful Camera is released under the MIT license. See `LICENSE` for more information.

            </div>

<div className={styles.card2}>
            <h2 className={styles.mdh2}> 📮 Contact</h2>

            If you have any questions or feedback about the Trustful Camera, please contact us at ssongaj@connect.ust.hk. We would love to hear from you!
            </div>

<div className={styles.card2}>
            <h2 className={styles.mdh2}>🎉 Acknowledgments</h2>

            The Trustful Camera project is based on the following open-source technologies:


            <a href="https://en.wikipedia.org/wiki/Blockchain">Blockchain technology</a> 
            <br />
            <a href="https://www.python.org/">Python</a> 
            <br />
            <a href="https://github.com/pallets/flask">Flask</a> 
            <br />
            <a href="https://github.com/ethereum/solidity">Solidity</a> 

          
          </div>

        </div>
      </main>

      <footer>
        <a
          href="https://hkust.edu.hk"
          target="_blank"
          rel="noopener noreferrer"
        >
          Powered by{' '}
          <img src="/hkust.png" alt="HKUST logo" className={styles.logo} />
        </a>
      </footer>

      <style jsx>{`
        main {
          padding: 5rem 0;
          flex: 1;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
        }
        footer {
          width: 100%;
          height: 100px;
          border-top: 1px solid #eaeaea;
          display: flex;
          justify-content: center;
          align-items: center;
        }
        footer img {
          margin-left: 0.5rem;
        }
        footer a {
          display: flex;
          justify-content: center;
          align-items: center;
          text-decoration: none;
          color: inherit;
        }
        code {
          background: #fafafa;
          border-radius: 5px;
          padding: 0.75rem;
          font-size: 1.1rem;
          font-family: Menlo, Monaco, Lucida Console, Liberation Mono,
            DejaVu Sans Mono, Bitstream Vera Sans Mono, Courier New, monospace;
        }
      `}</style>

      <style jsx global>{`
        html,
        body {
          padding: 0;
          margin: 0;
          font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto,
            Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue,
            sans-serif;
        }
        * {
          box-sizing: border-box;
        }
      `}</style>
    </div>
  );
};
export default Contact;