USE CompanyX


-- export.sql
;WITH cust_metrics AS (
    SELECT
        soh.CustomerID,
        COUNT(*) AS Purchases,
        AVG(sod.LineTotal) AS AvgOrderValue,
        SUM(CASE WHEN sod.SpecialOfferID <> 1 THEN 1 ELSE 0 END)*1.0 / COUNT(*) AS PromoResponseRate,
        DATEDIFF(day, MAX(soh.OrderDate), GETDATE()) AS Recency
    FROM Sales.SalesOrderHeader soh
    JOIN Sales.SalesOrderDetail sod ON soh.SalesOrderID = sod.SalesOrderID
    GROUP BY soh.CustomerID
)
SELECT
    sod.SalesOrderDetailID AS ProposalID,
    sod.LineTotal          AS ProposalAmount,
    p.StandardCost * sod.OrderQty AS Cost,
    st.Name                AS Region,
    CASE WHEN c.StoreID IS NULL THEN 'Individual' ELSE 'Store' END AS CustomerType,
    pc.Name                AS ProductCategory,
    CASE WHEN soh.Status IN (5) THEN 1 WHEN soh.Status IN (4,6) THEN 0 ELSE NULL END AS Accepted, -- shipped=1, rejected/cancelled=0
    cm.Purchases,
    cm.AvgOrderValue,
    cm.PromoResponseRate,
    cm.Recency
FROM Sales.SalesOrderDetail sod
JOIN Sales.SalesOrderHeader soh ON soh.SalesOrderID = sod.SalesOrderID
LEFT JOIN Sales.Customer c       ON soh.CustomerID = c.CustomerID
LEFT JOIN Sales.SalesTerritory st ON soh.TerritoryID = st.TerritoryID
LEFT JOIN Production.Product p    ON sod.ProductID = p.ProductID
LEFT JOIN Production.ProductSubcategory psc ON p.ProductSubcategoryID = psc.ProductSubcategoryID
LEFT JOIN Production.ProductCategory pc     ON psc.ProductCategoryID = pc.ProductCategoryID
LEFT JOIN cust_metrics cm ON cm.CustomerID = soh.CustomerID
WHERE soh.Status IN (4,5,6); -- keep rows with clear accept/reject/cancel status
