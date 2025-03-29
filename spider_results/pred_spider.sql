select count(*) from singer
select name from instructor where id in (select distinct t1.id from teaches as t1 join professor as t2 on t1.id = t2.id)
select name , country , age from singer order by age desc
select name , country , age from singer order by age desc
select avg(age) , min(age) , max(age) from singer where country = "france"
select avg(age) , min(age) , max(age) from singer where country = "france"
select t2.title , t2.year from singer as t1 join song as t2 on t1.singer_id = t2.singer_id order by t1.age limit 1
select t2.name , t2.year from singer as t1 join song as t2 on t1.singer_id = t2.singer_id where t1.age = (select min(age) from singer)
select distinct country from singer where age > 20
select distinct country from singer where age > 20
