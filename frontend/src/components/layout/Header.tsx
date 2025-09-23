import { BellIcon, UserCircleIcon, Bars3Icon } from '@heroicons/react/24/outline';
import Button from '../ui/Button';

interface HeaderProps {
  onMobileMenuClick: () => void;
}

const Header = ({ onMobileMenuClick }: HeaderProps) => {
  return (
    <header className="bg-white shadow-sm border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center py-4">
          {/* Left side - mobile menu button and title */}
          <div className="flex items-center">
            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden mr-3"
              onClick={onMobileMenuClick}
            >
              <Bars3Icon className="h-6 w-6" />
            </Button>
            
            <h1 className="text-2xl font-semibold text-gray-900">
              Security Dashboard
            </h1>
          </div>
          
          {/* Right side - notifications and user menu */}
          <div className="flex items-center space-x-4">
            {/* Notifications */}
            <Button variant="ghost" size="sm" className="relative">
              <BellIcon className="h-6 w-6" />
              <span className="absolute -top-1 -right-1 h-4 w-4 bg-red-500 rounded-full text-xs text-white flex items-center justify-center">
                3
              </span>
            </Button>
            
            {/* User menu */}
            <Button variant="ghost" size="sm">
              <UserCircleIcon className="h-8 w-8" />
            </Button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;