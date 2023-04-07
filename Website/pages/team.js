import React from "react";
import styles from '../styles/Home.module.css';


const Team = () => {
  return (
    <div className={styles.container}>
      <main>
        <h1 className={styles.teamHead}>
          Meet our <a href="https://nextjs.org">Team</a>
        </h1>

        <p className={styles.description}>
          <code>A team of ddl chasers @ HKUST.</code>
        </p>

        <div className={styles.grid2}>
          <a href="/" className={styles.card2}>
            <h3>CHANG Tianxing</h3>
          </a>

          <a href="/" className={styles.card2}>
            <h3>SONG Shiyuan</h3>
          </a>

          <a href="/" className={styles.card2}>
            <h3>ZHANG Liyu </h3>
          </a>

          <a href="/" className={styles.card2}>
            <h3>ZHOU Sitian </h3>
          </a>

          <a href="/" className={styles.card2}>
            <h3>ZONG Haosong </h3>
          </a>

          <a href="/" className={styles.card2}>
            <h3> Yan Xizhi</h3>
          </a>

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

export default Team;