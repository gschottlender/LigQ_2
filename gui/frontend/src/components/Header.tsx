import { CircleQuestionMark, Moon, Search, Settings, Sun } from "lucide-react";
import { useState } from "react";
import { NavLink } from "react-router-dom";

export function Header(){
    const navLinkClass = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-2 px-5 py-2 rounded-lg font-semibold font-montserrat transition-all duration-300
    ${isActive ? 'bg-white dark:bg-gray-500 text-gray-500 dark:text-white shadow-xs' : 'text-gray-500 dark:text-gray-400'}`

    const [isDark, setIsDark] = useState(() => {
        return document.documentElement.classList.contains('dark');
    });
    const toggleDark = () => {
        setIsDark(!isDark);
        document.documentElement.classList.toggle('dark')
    }

    return(
        <header className="h-20 px-6 py-4 border-b border-gray-300 flex justify-between dark:text-white dark:bg-[#1a2330]">
            <div className="flex items-center gap-2">
                <img src="/favicon.svg" className="w-10 h-10"/>
                <p className="text-2xl font-semibold font-dm-sans"> LigQ_2 </p>
            </div>

            <nav className="bg-gray-200 dark:bg-gray-700 p-1 rounded-[10px] flex items-center gap-2">
                <NavLink to="/configure" className={navLinkClass}>
                    <Settings className="w-4 h-4"/>
                    <p className="text-sm font-dm-sans"> Configure Search</p>
                </NavLink>

                <NavLink to="/" className={navLinkClass}>
                    <Search className="w-4 h-4 hover:text-gray-400"/>
                    <p className="text-sm font-dm-sans">Visualize Results</p>
                </NavLink>
            </nav>
            <div className="flex items-center gap-4">
                <NavLink to="/help" className="flex items-center gap-2 cursor-pointer transition-colors duration-200 hover:text-gray-300 group">
                    <CircleQuestionMark className="w-5 h-5 text-gray-500 group-hover:text-gray-300 transition-colors duration-200 dark:text-gray-100" />
                    <p className="text-sm text-gray-500 font-dm-sans group-hover:text-gray-300 transition-colors duration-200 dark:text-gray-100"> Help </p>
                </NavLink>

                <button onClick={toggleDark} className="flex items-center gap-2 border border-gray-300 dark:border-gray-100 p-2.5 rounded-xl cursor-pointer transition-colors duration-200 group">
                    {isDark 
                        ? <Sun className="w-4 h-4 text-gray-100 group-hover:text-gray-300 transition-colors duration-200" />
                        : <Moon className="w-4 h-4 text-gray-500 group-hover:text-gray-300 transition-colors duration-200" />
                    }
                </button>
            </div>
        </header>
    );
}