import { Route, Routes } from 'react-router-dom';
import { AuthenticationGuard } from './components/Auth/Auth';
import Home from './Home';
import { SKIP_AUTH } from './utils/Constants.ts';
import AgenticChatPage from './components/AgenticChat/AgenticChatPage';

const App = () => {
  return (
    <Routes>
      <Route path='/' element={<AgenticChatPage />}></Route>
      <Route path='/chat-only' element={<AgenticChatPage />}></Route>
      <Route path='/builder' element={SKIP_AUTH ? <Home /> : <AuthenticationGuard component={Home} />}></Route>
      <Route path='/readonly' element={<Home />}></Route>
    </Routes>
  );
};
export default App;
