import Link from "next/link";
import styles from '../styles/Home.module.css';

const NavItem = ({ text, href, active }) => {
  return (
    <Link href={href} className={styles.nav_link}>
      {text}
    </Link>
  );
};

export default NavItem;