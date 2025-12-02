USE CompanyX

SELECT * 
FROM sys.tables 

-- Promotions per year & month
SELECT 
    YEAR(StartDate) AS year,
    MONTH(StartDate) AS month,
    COUNT(*) AS num_promotions
FROM Sales.SpecialOffer
GROUP BY YEAR(StartDate), MONTH(StartDate)
ORDER BY year, month;

SELECT
    MONTH(StartDate) AS month,
    SUM(CASE WHEN YEAR(StartDate) = 2011 THEN 1 ELSE 0 END) AS promotions_2011,
    SUM(CASE WHEN YEAR(StartDate) = 2012 THEN 1 ELSE 0 END) AS promotions_2012,
    SUM(CASE WHEN YEAR(StartDate) = 2013 THEN 1 ELSE 0 END) AS promotions_2013,
    SUM(CASE WHEN YEAR(StartDate) = 2014 THEN 1 ELSE 0 END) AS promotions_2014
FROM Sales.SpecialOffer
GROUP BY MONTH(StartDate)
ORDER BY month;

-- Revenue comparison with vs without promotion
SELECT
    CASE WHEN sod.SpecialOfferID = 1 THEN 'No Promotion' ELSE 'Promotion' END AS promo_flag,
    SUM(sod.LineTotal) AS total_revenue
FROM Sales.SalesOrderDetail sod
GROUP BY CASE WHEN sod.SpecialOfferID = 1 THEN 'No Promotion' ELSE 'Promotion' END;
 --Total Profit--
SELECT
    CASE WHEN sod.SpecialOfferID = 1 THEN 'No Promotion' ELSE 'Promotion' END AS promo_flag,
    SUM((sod.UnitPrice - p.StandardCost) * sod.OrderQty) AS total_profit
FROM Sales.SalesOrderDetail sod
JOIN Production.Product p ON sod.ProductID = p.ProductID
GROUP BY CASE WHEN sod.SpecialOfferID = 1 THEN 'No Promotion' ELSE 'Promotion' END;

SELECT
    CASE WHEN sod.SpecialOfferID = 1 THEN 'No Promotion' ELSE 'Promotion' END AS promo_flag,
    SUM((sod.UnitPrice - p.StandardCost) * sod.OrderQty) * 100.0 / SUM(sod.LineTotal) AS profit_margin_percent
FROM Sales.SalesOrderDetail sod
JOIN Production.Product p ON sod.ProductID = p.ProductID
GROUP BY CASE WHEN sod.SpecialOfferID = 1 THEN 'No Promotion' ELSE 'Promotion' END;

-- Best-selling products under promotions
SELECT TOP 10 
    p.Name AS product_name,
    SUM(sod.OrderQty) AS total_quantity,
    SUM(sod.LineTotal) AS total_revenue
FROM Sales.SalesOrderDetail sod
JOIN Production.Product p ON sod.ProductID = p.ProductID
WHERE sod.SpecialOfferID <> 1   -- exclude "No Discount"
GROUP BY p.Name
ORDER BY total_quantity DESC;

SELECT
    YEAR(soh.OrderDate) AS order_year,
    SUM(CASE WHEN sod.SpecialOfferID = 1 THEN sod.LineTotal ELSE 0 END) AS no_promotion_sales,
    SUM(CASE WHEN sod.SpecialOfferID <> 1 THEN sod.LineTotal ELSE 0 END) AS promotion_sales
FROM Sales.SalesOrderHeader soh
JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
GROUP BY YEAR(soh.OrderDate)
ORDER BY order_year;

----------- limitation

SELECT
    COUNT(*) AS total_products,
    SUM(CASE WHEN Weight IS NULL OR Weight = 0 THEN 1 ELSE 0 END) AS missing_weight,
    SUM(CASE WHEN Size IS NULL THEN 1 ELSE 0 END) AS missing_size,
    SUM(CASE WHEN Style IS NULL OR Style = 'NA' THEN 1 ELSE 0 END) AS missing_style
FROM Production.Product;


SELECT
    pc.Name AS category,
    SUM(sod.LineTotal) AS total_revenue
FROM Sales.SalesOrderDetail sod
JOIN Production.Product p ON sod.ProductID = p.ProductID
JOIN Production.ProductSubcategory psc ON p.ProductSubcategoryID = psc.ProductSubcategoryID
JOIN Production.ProductCategory pc ON psc.ProductCategoryID = pc.ProductCategoryID
GROUP BY pc.Name
ORDER BY total_revenue DESC;

SELECT
    pc.Name AS category,
    COUNT(sod.SalesOrderDetailID) AS num_line_items
FROM Sales.SalesOrderDetail sod
JOIN Production.Product p ON sod.ProductID = p.ProductID
JOIN Production.ProductSubcategory psc ON p.ProductSubcategoryID = psc.ProductSubcategoryID
JOIN Production.ProductCategory pc ON psc.ProductCategoryID = pc.ProductCategoryID
GROUP BY pc.Name
ORDER BY num_line_items DESC;

SELECT TOP 10
    soh.OrderDate,
    p.Name AS product_name,
    sod.OrderQty,
    sod.UnitPrice,
    sod.LineTotal
FROM Sales.SalesOrderDetail sod
JOIN Sales.SalesOrderHeader soh ON sod.SalesOrderID = soh.SalesOrderID
JOIN Production.Product p ON sod.ProductID = p.ProductID
ORDER BY sod.LineTotal DESC;