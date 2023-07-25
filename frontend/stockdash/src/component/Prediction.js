import React, { useState, useEffect, useContext } from "react";
import Card from "./card";
import axios from "axios";
import StockContext from "../context/StockContext";
import { fetchQuote, fetchStockDetails } from "../API/StockApi";
import ThemeContext from "../context/ThemeContext";


function Prediction() {
    const [data, setData] = useState([{}]);
    const { stockSymbol } = useContext(StockContext);
    const [stockDetails, setstockDetails] = useState(ThemeContext);
    const [quote, setQuote] = useState({});

    useEffect(() => {


        const updateStockDetails = async () => {
            try {
                const result = await fetchStockDetails(stockSymbol);
                setstockDetails(result);

            }
            catch (error) {
                setstockDetails({});
                console.log(error);
            }
        };
        const updateStockOverview = async () => {
            try {
                const result = await fetchQuote(stockSymbol);
                setQuote(result);
            }
            catch (error) {
                setQuote({});
                console.log(error);
            }
        };

        const fetchPredictionData = async () => {
            try {
                // Use stockSymbol in the request URL
                const response = await axios.post('/predict', { stockSymbol }, {
                    headers: {
                        "Content-Type": "application/json", // Set the Content-Type to JSON
                    },
                });
                setData(response.data);
                console.log(response.data);
            } catch (error) {
                setData([]);
                console.log(error);
            }
        };

        fetchPredictionData();
        updateStockDetails();
        updateStockOverview();
    }, [stockSymbol]);
    return (
        <Card>

            {
                 
                <ul>
                    <span className="absolute left-1 top-1 text-neutral-400 text-lg xl:text-xl 2xl:text-2xl">Prediction</span> 
                    {data.map((predictionObj, index) => (
                        <li key={index}>
                            <div className="w-full h-full flex items-baseline justify-center">
                            <span className="absolute left-5 text-neutral-400 text-lg xl:text-xl 2xl:text-2xl pt-2 pb-6">{predictionObj.date}</span>
                            
                                <span className="absolute top-1 text-2xl xl:text-4xl 2xl:text-6xl flex items-center pt-8 pb-12">${parseFloat(predictionObj.prediction).toFixed(2)}
                                    <span className="text-lg xl:text-xl 2xl:text-2xl text-neutral-400 m-2">
                                        {stockDetails.currency} </span>
                                </span>
                                <span className={`absolute right-1 top-1 text-lg xl:text-xl 2xl:text-2xl pt-12 pb-4 ${parseFloat(predictionObj.prediction - quote.pc).toFixed(2) > 0 ? "text-lime-500" : "text-red-500"
                                    }`}>  {parseFloat(predictionObj.prediction - quote.pc).toFixed(2)}
                                    <span>({parseFloat((predictionObj.prediction - quote.pc) / quote.pc * 100).toFixed(4)}%)</span>
                                </span>
                            </div>
                        </li>
                    ))}
                </ul>
            }

        </Card>
    );
};



export default Prediction