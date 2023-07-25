import React, { useState } from "react";
import Card from "./card";
import ThemeContext from "../context/ThemeContext";

const Detail = ({detail}) => {
    const {darkMode} = useState(ThemeContext);
    const detailList={
        name: "Name",
        country: "Country",
        currency: "Currency",
        exchange: "Exchange",
        ipo: "IPO Date",
        marketCapitalization: "Market Capitalization",
        finnhubIndustry: "Industry",
        
    };
    const convertMillionToBillion = (number) =>{
        return (number/1000).toFixed(2);
    }
  return (
    <Card>
        <ul className={`w-full h-full flex flex-col justify-between divide-y-1 ${darkMode ? "divide-gray-800" : null}`}>
            {Object.keys(detailList).map((item)=>{
                return <li key={item}
                className="flex-1 flex justify-between item-center">
                    <span>{detailList[item]}</span>
                    <span>{item==="marketCapitalization"
                    ? `${convertMillionToBillion(detail[item])}B` 
                    : detail[item]}</span>
                </li>
            })}
        </ul>
    </Card>
  );
}

export default Detail