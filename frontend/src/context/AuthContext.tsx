import React, { createContext, useState, useContext, useEffect } from 'react';

interface AuthContextType {
  isLoggedIn: boolean;
  userName: string;
  login: (name: string) => void;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType>({
  isLoggedIn: false,
  userName: '',
  login: () => {},
  logout: () => {},
});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState<boolean>(() => {
    return localStorage.getItem('physio_logged_in') === 'true';
  });
  const [userName, setUserName] = useState<string>(() => {
    return localStorage.getItem('physio_user_name') || 'Student';
  });

  // No longer need useEffect for initial load since it's synchronous now


  const login = (name: string) => {
    setIsLoggedIn(true);
    setUserName(name);
    localStorage.setItem('physio_logged_in', 'true');
    localStorage.setItem('physio_user_name', name);
  };

  const logout = () => {
    setIsLoggedIn(false);
    setUserName('');
    localStorage.removeItem('physio_logged_in');
    localStorage.removeItem('physio_user_name');
  };

  return (
    <AuthContext.Provider value={{ isLoggedIn, userName, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
};
