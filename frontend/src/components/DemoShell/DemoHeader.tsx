import { ReactNode } from 'react';
import { NavLink } from 'react-router-dom';
import './DemoHeader.css';

type Props = {
  right?: ReactNode;
};

export default function DemoHeader({ right }: Props) {
  return (
    <header className='demo-header'>
      <div className='demo-header-inner'>
        <div className='demo-brand'>
          <div className='demo-title'>Nornikel Agentic RAG</div>
          <div className='demo-subtitle'>Научный клубок</div>
        </div>
        <nav className='demo-nav' aria-label='Основная навигация'>
          <NavLink to='/' end className={({ isActive }) => isActive ? 'demo-nav-link active' : 'demo-nav-link'}>
            Чат
          </NavLink>
          <NavLink to='/graph' className={({ isActive }) => isActive ? 'demo-nav-link active' : 'demo-nav-link'}>
            Граф
          </NavLink>
          <NavLink to='/data' className={({ isActive }) => isActive ? 'demo-nav-link active' : 'demo-nav-link'}>
            Данные
          </NavLink>
        </nav>
        <div className='demo-header-action'>{right}</div>
      </div>
    </header>
  );
}
