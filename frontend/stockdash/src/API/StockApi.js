const BASE_URL="https://finnhub.io/api/v1/";
const Key="&token=cip5aphr01qrdahju3f0cip5aphr01qrdahju3fg";


export const searchSymbols = async (query) =>{
    const url= `${BASE_URL}search?q=${query}${Key}`;
    const response = await fetch(url);

    if (!response.ok){
        const message = `An error has occured: ${response.status}`;
        throw new Error(message);
    }
    return await response.json();
};

export const fetchStockDetails = async (stockSymbol) =>{
    const url = `${BASE_URL}stock/profile2?symbol=${stockSymbol}${Key}`;
    const response = await fetch(url);

    if (!response.ok){
        const message = `An error has occured: ${response.status}`;
        throw new Error(message);
    }
    return await response.json();
};

export const fetchQuote = async (stockSymbol) => {
    const url = `${BASE_URL}quote?symbol=${stockSymbol}${Key}`; 
    const response = await fetch(url);

    if (!response.ok){
        const message = `An error has occured: ${response.status}`;
        throw new Error(message);
    }
    return await response.json();
};

export const fetchHistoricalData = async (
    stockSymbol,
    resolution,
    from,
    to
 ) => {
    const url =`${BASE_URL}stock/candle?symbol=${stockSymbol}&resolution=${resolution}&from=${from}&to=${to}${Key}`;
    const response = await fetch(url);

    if (!response.ok){
        const message = `An error has occured: ${response.status}`;
        throw new Error(message);
    }
    return await response.json();
 };
