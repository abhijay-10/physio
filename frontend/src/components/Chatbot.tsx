import React, { useState, useRef, useEffect } from 'react';
import { X, Send, MessageCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { useLocation } from 'react-router-dom';
import './Chatbot.css';
import chatbotIcon from '../assets/chatbot_icon_new.png';

interface Message {
  id: string;
  text: string;
  sender: 'user' | 'bot';
}

const Chatbot: React.FC = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { id: '1', text: 'Hi! I\'m your PhysioMaster Assistant. I can help you navigate diagnostic modules, explain AI precision scores, or guide you through proper patient positioning. What do you need help with?', sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [showTooltip, setShowTooltip] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  
  const location = useLocation();

  useEffect(() => {
    if (isOpen) {
      setShowTooltip(false);
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [isOpen, messages, isTyping]);

  const handleSendMessage = async (text: string) => {
    if (!text.trim()) return;
    
    const userMsg: Message = { id: Date.now().toString(), text, sender: 'user' };
    setMessages(prev => [...prev, userMsg]);
    setInputValue("");
    setIsTyping(true);
    
    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          current_page: location.pathname,
          history: messages
        })
      });
      
      const data = await response.json();
      const botMsg: Message = { id: (Date.now() + 1).toString(), text: data.response, sender: 'bot' };
      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      const errorMsg: Message = { 
        id: (Date.now() + 1).toString(), 
        text: "I'm having trouble connecting to the server. Please ensure the backend is running.", 
        sender: 'bot' 
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const getSuggestedPrompts = () => {
    const path = location.pathname;
    if (path.includes('/chest')) return ["How do I align for this Chest view?", "What does the Precision Score mean?"];
    if (path.includes('/hand')) return ["What is a PA Hand vs Oblique Hand?", "Why is it failing to detect the wrist?"];
    if (path.includes('/spine')) return ["How do you measure Kyphosis?", "How should the patient stand?"];
    return ["What is PhysioMaster?", "How does the AI positioning work?", "Which X-Ray views are supported?"];
  };

  return (
    <div className="chatbot-container">
      {/* Tooltip that shows initially */}
      {!isOpen && showTooltip && (
        <div className="chatbot-tooltip" onClick={() => setIsOpen(true)}>
          Hi, How may I assist you today?
        </div>
      )}

      {/* The Chat Window */}
      {isOpen && (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <div className="chatbot-header-info">
              <img src={chatbotIcon} alt="Fox Avatar" className="chatbot-avatar-small" />
              <span>Physio Assistant</span>
            </div>
            <button className="chatbot-close-btn" onClick={() => setIsOpen(false)}>
              <X size={20} />
            </button>
          </div>
          
          <div className="chatbot-messages">
            {messages.map((msg) => (
              <div key={msg.id} className={`chatbot-message-wrapper ${msg.sender}`}>
                {msg.sender === 'bot' && <img src={chatbotIcon} alt="Bot" className="chatbot-message-avatar" />}
                <div className={`chatbot-message ${msg.sender}`}>
                  {msg.sender === 'bot' ? (
                    <ReactMarkdown>{msg.text}</ReactMarkdown>
                  ) : (
                    msg.text
                  )}
                </div>
              </div>
            ))}
            {isTyping && (
              <div className="chatbot-message-wrapper bot">
                <img src={chatbotIcon} alt="Bot" className="chatbot-message-avatar" />
                <div className="chatbot-message bot typing-indicator">
                  <span></span><span></span><span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          <div className="chatbot-input-area">
            {messages.length === 1 && (
              <div className="chatbot-options-list horizontal">
                {getSuggestedPrompts().map((q, idx) => (
                  <button 
                    key={idx} 
                    className="chatbot-option-chip"
                    onClick={() => handleSendMessage(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
            <form 
              className="chatbot-input-form"
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage(inputValue);
              }}
            >
              <input 
                type="text" 
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                placeholder="Ask me anything..." 
                className="chatbot-input"
              />
              <button type="submit" className="chatbot-send-btn" disabled={!inputValue.trim() || isTyping}>
                <Send size={18} />
              </button>
            </form>
          </div>
        </div>
      )}

      {/* The Floating Action Button */}
      <div className="chatbot-fab" onClick={() => setIsOpen(!isOpen)}>
        {isOpen ? (
          <X size={28} color="white" />
        ) : (
          <img src={chatbotIcon} alt="Chat" className="chatbot-fab-image" />
        )}
      </div>
    </div>
  );
};

export default Chatbot;
