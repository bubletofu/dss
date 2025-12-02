-- Params
DECLARE @FromDate date = '2011-01-01';
DECLARE @ToDate   date = '2014-12-31';

--------------------------------------------
-- 0) Build reusable temp tables
--------------------------------------------
IF OBJECT_ID('tempdb..#line_base') IS NOT NULL DROP TABLE #line_base;

SELECT
    soh.SalesOrderID,
    CAST(soh.OrderDate AS date)                                        AS OrderDate,
    DATEFROMPARTS(YEAR(soh.OrderDate), MONTH(soh.OrderDate), 1)        AS MonthStart,
    YEAR(soh.OrderDate)                                                AS [year],
    MONTH(soh.OrderDate)                                               AS [month],

    soh.TerritoryID,
    st.Name                                                            AS TerritoryName,

    sod.ProductID,
    p.Name                                                             AS ProductName,
    p.ProductNumber,
    pc.ProductCategoryID,
    pc.Name                                                            AS CategoryName,
    psc.ProductSubcategoryID,
    psc.Name                                                           AS SubcategoryName,

    sod.SpecialOfferID,
    CASE WHEN sod.SpecialOfferID = 1 OR sod.UnitPriceDiscount = 0 THEN 0 ELSE 1 END AS IsPromotion,  -- robust
    sod.OrderQty,
    sod.UnitPrice,
    sod.UnitPriceDiscount,   -- (0..1) in AdventureWorks

    (sod.UnitPrice * sod.OrderQty)                                     AS GrossBeforeDiscount,
    (sod.UnitPrice * sod.UnitPriceDiscount * sod.OrderQty)             AS DiscountAmount,
    sod.LineTotal                                                      AS NetRevenue  -- = gross - discount
INTO #line_base
FROM Sales.SalesOrderHeader  AS soh
JOIN Sales.SalesOrderDetail  AS sod ON soh.SalesOrderID = sod.SalesOrderID
LEFT JOIN Sales.SalesTerritory st    ON soh.TerritoryID = st.TerritoryID
JOIN Production.Product       p      ON sod.ProductID = p.ProductID
LEFT JOIN Production.ProductSubcategory psc ON p.ProductSubcategoryID = psc.ProductSubcategoryID
LEFT JOIN Production.ProductCategory    pc  ON psc.ProductCategoryID = pc.ProductCategoryID
WHERE soh.OrderDate >= @FromDate
  AND soh.OrderDate < DATEADD(day, 1, @ToDate);

-- (optional) tiny index for speed
CREATE CLUSTERED INDEX IX_line_base ON #line_base (MonthStart, IsPromotion);

-- Order-level promo flag (each order in exactly one bucket)
IF OBJECT_ID('tempdb..#order_flag') IS NOT NULL DROP TABLE #order_flag;
SELECT SalesOrderID, MAX(IsPromotion) AS OrderHasPromo
INTO #order_flag
FROM #line_base
GROUP BY SalesOrderID;

CREATE CLUSTERED INDEX IX_order_flag ON #order_flag (SalesOrderID);

--------------------------------------------
-- 1) Promotions launched per year/month (exclude ID = 1 "No Discount")
--------------------------------------------
SELECT
  YEAR(so.StartDate)  AS [year],
  MONTH(so.StartDate) AS [month],
  COUNT(*)            AS num_promotions_launched
FROM Sales.SpecialOffer AS so
WHERE so.SpecialOfferID <> 1
  AND so.StartDate >= @FromDate
  AND so.StartDate <= @ToDate
GROUP BY YEAR(so.StartDate), MONTH(so.StartDate)
ORDER BY [year], [month];

--------------------------------------------
-- 1b) Active promotions by month
--------------------------------------------
;WITH months AS (
  SELECT DATEFROMPARTS(YEAR(@FromDate), MONTH(@FromDate), 1) AS MonthStart
  UNION ALL
  SELECT DATEADD(month, 1, MonthStart)
  FROM months
  WHERE DATEADD(month, 1, MonthStart) <= DATEFROMPARTS(YEAR(@ToDate), MONTH(@ToDate), 1)
)
SELECT
  YEAR(m.MonthStart)  AS [year],
  MONTH(m.MonthStart) AS [month],
  COUNT(DISTINCT so.SpecialOfferID) AS active_promotions
FROM months m
JOIN Sales.SpecialOffer so
  ON so.SpecialOfferID <> 1
 AND so.StartDate <= EOMONTH(m.MonthStart)
 AND COALESCE(so.EndDate, '9999-12-31') >= m.MonthStart
GROUP BY YEAR(m.MonthStart), MONTH(m.MonthStart)
ORDER BY [year], [month]
OPTION (MAXRECURSION 0);

--------------------------------------------
-- 2) Sales with vs without promotion (monthly, line level)
--------------------------------------------
;WITH monthly_line AS (
  SELECT
    lb.MonthStart,
    lb.IsPromotion,
    SUM(lb.NetRevenue)          AS Revenue,
    SUM(lb.OrderQty)            AS Units,
    SUM(lb.DiscountAmount)      AS DiscountTotal,
    SUM(lb.GrossBeforeDiscount) AS GrossBeforeDiscount
  FROM #line_base lb
  GROUP BY lb.MonthStart, lb.IsPromotion
),
tot AS (
  SELECT MonthStart, SUM(Revenue) AS TotalRevenue
  FROM monthly_line
  GROUP BY MonthStart
)
SELECT
  YEAR(m.MonthStart)  AS [year],
  MONTH(m.MonthStart) AS [month],
  CASE WHEN m.IsPromotion = 1 THEN 'Promotion' ELSE 'No Promotion' END AS promo_flag,
  m.Revenue,
  m.Units,
  CAST(NULLIF(m.Revenue,0) * 1.0 / NULLIF(m.Units,0) AS decimal(18,2)) AS ASP,
  m.DiscountTotal,
  CAST(m.Revenue * 1.0 / NULLIF(t.TotalRevenue,0) AS decimal(18,4))    AS RevenueShare
FROM monthly_line m
JOIN tot t ON t.MonthStart = m.MonthStart
ORDER BY [year], [month], promo_flag;

--------------------------------------------
-- 2b) Orders with vs without promotion (no double-count)
--------------------------------------------
;WITH orders_month AS (
  SELECT lb.SalesOrderID, lb.MonthStart
  FROM #line_base lb
  GROUP BY lb.SalesOrderID, lb.MonthStart
),
bucket AS (
  SELECT om.MonthStart, ofg.OrderHasPromo, COUNT(*) AS orders
  FROM orders_month om
  JOIN #order_flag ofg ON ofg.SalesOrderID = om.SalesOrderID
  GROUP BY om.MonthStart, ofg.OrderHasPromo
)
SELECT
  CONVERT(char(7), MonthStart, 126) AS YearMonth,
  SUM(CASE WHEN OrderHasPromo = 1 THEN orders ELSE 0 END) AS Orders_Promo,
  SUM(CASE WHEN OrderHasPromo = 0 THEN orders ELSE 0 END) AS Orders_NoPromo,
  SUM(orders) AS Orders_Total,
  CAST(SUM(CASE WHEN OrderHasPromo = 1 THEN orders ELSE 0 END)*1.0 / NULLIF(SUM(orders),0) AS decimal(5,2)) AS Promo_Order_Share
FROM bucket
GROUP BY MonthStart
ORDER BY MonthStart;

--------------------------------------------
-- 3) Top product groups during promotions (per month)
--------------------------------------------
DECLARE @TopN int = 5;

;WITH promo_sales AS (
  SELECT
    lb.MonthStart,
    COALESCE(pc.Name, 'Uncategorized')  AS CategoryName,
    COALESCE(psc.Name,'Unspecified')    AS SubcategoryName,
    SUM(lb.NetRevenue)                  AS Revenue,
    SUM(lb.OrderQty)                    AS Units
  FROM #line_base lb
  LEFT JOIN Production.ProductSubcategory psc ON lb.ProductSubcategoryID = psc.ProductSubcategoryID
  LEFT JOIN Production.ProductCategory    pc  ON lb.ProductCategoryID    = pc.ProductCategoryID
  WHERE lb.IsPromotion = 1
  GROUP BY lb.MonthStart, pc.Name, psc.Name
),
ranked AS (
  SELECT *,
         RANK() OVER (PARTITION BY MonthStart ORDER BY Revenue DESC) AS rnk
  FROM promo_sales
)
SELECT
  YEAR(MonthStart)  AS [year],
  MONTH(MonthStart) AS [month],
  CategoryName, SubcategoryName,
  Revenue, Units, rnk
FROM ranked
WHERE rnk <= @TopN
ORDER BY [year], [month], rnk;