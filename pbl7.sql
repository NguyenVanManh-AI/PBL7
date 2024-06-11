-- --------------------------------------------------------
-- Host:                         127.0.0.1
-- Server version:               8.0.30 - MySQL Community Server - GPL
-- Server OS:                    Win64
-- HeidiSQL Version:             12.1.0.6537
-- --------------------------------------------------------

/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET NAMES utf8 */;
/*!50503 SET NAMES utf8mb4 */;
/*!40103 SET @OLD_TIME_ZONE=@@TIME_ZONE */;
/*!40103 SET TIME_ZONE='+00:00' */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;

-- Dumping structure for table ai.admins
CREATE TABLE IF NOT EXISTS `admins` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `email` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `password` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `remember_token` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `admins_email_unique` (`email`)
) ENGINE=InnoDB AUTO_INCREMENT=2 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.admins: ~0 rows (approximately)
INSERT INTO `admins` (`id`, `email`, `password`, `remember_token`, `created_at`, `updated_at`) VALUES
	(1, 'admin123@gmail.com', '$2y$10$OYIeNC24UBracwkYdKfL3euyDfRC/uri3gaazeMSdSNLaHWH2dr/y', NULL, '2024-04-27 07:28:56', '2024-04-27 07:28:56');

-- Dumping structure for table ai.broadcasts
CREATE TABLE IF NOT EXISTS `broadcasts` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `line_user_id` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `title` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `sent_at` timestamp NOT NULL,
  `status` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `is_delete` tinyint(1) NOT NULL,
  `sender_id` bigint unsigned NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `broadcasts_sender_id_foreign` (`sender_id`),
  CONSTRAINT `broadcasts_sender_id_foreign` FOREIGN KEY (`sender_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.broadcasts: ~0 rows (approximately)

-- Dumping structure for table ai.channels
CREATE TABLE IF NOT EXISTS `channels` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `account_manager_id` bigint unsigned NOT NULL,
  `channel_id` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `channel_name` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `channel_secret` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `channel_access_token` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `picture_url` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `channels_account_manager_id_foreign` (`account_manager_id`),
  CONSTRAINT `channels_account_manager_id_foreign` FOREIGN KEY (`account_manager_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.channels: ~2 rows (approximately)
INSERT INTO `channels` (`id`, `account_manager_id`, `channel_id`, `channel_name`, `channel_secret`, `channel_access_token`, `picture_url`, `created_at`, `updated_at`) VALUES
	(1, 1, '2003175796', '[test]chanel_test1', '779f329a5ba363654bb7edbc2b84b6d7', 'Sgxs8tgbGJN41Q92WuIQPIggm3feQjyK3GWGClLc0t16jPhg15HXsECa4l/ce39y54Vpd2PmoxcthCJZyAjYlUeoivpkWtVeXjQQ5vGpEpa7uo2bm2j8exh4zrzIG6C7yJacZtcN1NRlOP7wYN0PnwdB04t89/1O/w1cDnyilFU=', NULL, '2024-04-27 07:28:56', '2024-04-27 07:28:56'),
	(2, 2, '2003308317', 'test_dut1', '1446e7d4c9eec2443fc21ce0cf4b45ec', 'txnQ8efyzbIGDv6VZ1HrATucI+rskmpJGYquRfuhTqUwnGude26UC75NIR5mtWRbyIJ6piDEaedbW3dPeJjL5Gmj0fns5EW1tDa7aXBXpH/hwDWR+KFYD0rggc3p8uplB8VIhcqSslKShINtK58JhgdB04t89/1O/w1cDnyilFU=', NULL, '2024-04-27 07:28:56', '2024-04-27 07:28:56');

-- Dumping structure for table ai.channel_statisticals
CREATE TABLE IF NOT EXISTS `channel_statisticals` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `channel_id` bigint unsigned NOT NULL,
  `date` date NOT NULL,
  `api_broadcast` int DEFAULT '0',
  `api_multicast` int DEFAULT '0',
  `followers` int DEFAULT '0',
  `targeted_reaches` int DEFAULT '0',
  `blocks` int DEFAULT '0',
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `channel_statisticals_channel_id_foreign` (`channel_id`),
  CONSTRAINT `channel_statisticals_channel_id_foreign` FOREIGN KEY (`channel_id`) REFERENCES `channels` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=63 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.channel_statisticals: ~62 rows (approximately)
INSERT INTO `channel_statisticals` (`id`, `channel_id`, `date`, `api_broadcast`, `api_multicast`, `followers`, `targeted_reaches`, `blocks`, `created_at`, `updated_at`) VALUES
	(1, 1, '2024-03-27', 84, 6, 7, 30, 21, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(2, 2, '2024-03-27', 71, 58, 2, 11, 3, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(3, 1, '2024-03-28', 37, 8, 29, 5, 16, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(4, 2, '2024-03-28', 6, 46, 30, 8, 30, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(5, 1, '2024-03-29', 58, 5, 22, 19, 1, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(6, 2, '2024-03-29', 74, 96, 10, 7, 16, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(7, 1, '2024-03-30', 87, 42, 29, 1, 22, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(8, 2, '2024-03-30', 13, 33, 16, 4, 29, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(9, 1, '2024-03-31', 37, 86, 5, 9, 30, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(10, 2, '2024-03-31', 91, 33, 12, 29, 27, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(11, 1, '2024-04-01', 19, 83, 26, 25, 23, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(12, 2, '2024-04-01', 90, 86, 11, 1, 9, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(13, 1, '2024-04-02', 20, 49, 16, 12, 17, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(14, 2, '2024-04-02', 86, 55, 19, 16, 13, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(15, 1, '2024-04-03', 5, 3, 14, 10, 14, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(16, 2, '2024-04-03', 96, 94, 20, 9, 27, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(17, 1, '2024-04-04', 25, 80, 27, 20, 14, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(18, 2, '2024-04-04', 24, 65, 17, 20, 24, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(19, 1, '2024-04-05', 36, 24, 1, 20, 3, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(20, 2, '2024-04-05', 100, 3, 2, 18, 13, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(21, 1, '2024-04-06', 59, 27, 0, 22, 26, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(22, 2, '2024-04-06', 10, 1, 24, 27, 25, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(23, 1, '2024-04-07', 30, 49, 16, 18, 8, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(24, 2, '2024-04-07', 86, 71, 30, 8, 4, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(25, 1, '2024-04-08', 89, 16, 10, 22, 12, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(26, 2, '2024-04-08', 59, 39, 28, 8, 23, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(27, 1, '2024-04-09', 55, 61, 28, 1, 4, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(28, 2, '2024-04-09', 15, 2, 5, 20, 26, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(29, 1, '2024-04-10', 84, 41, 0, 11, 23, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(30, 2, '2024-04-10', 37, 82, 13, 26, 23, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(31, 1, '2024-04-11', 78, 0, 4, 4, 21, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(32, 2, '2024-04-11', 35, 67, 24, 29, 0, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(33, 1, '2024-04-12', 15, 24, 8, 9, 22, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(34, 2, '2024-04-12', 12, 56, 8, 22, 2, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(35, 1, '2024-04-13', 100, 69, 27, 28, 24, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(36, 2, '2024-04-13', 18, 13, 1, 14, 8, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(37, 1, '2024-04-14', 29, 100, 6, 22, 3, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(38, 2, '2024-04-14', 54, 9, 5, 17, 9, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(39, 1, '2024-04-15', 8, 3, 25, 9, 14, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(40, 2, '2024-04-15', 57, 70, 23, 17, 1, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(41, 1, '2024-04-16', 82, 65, 24, 9, 1, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(42, 2, '2024-04-16', 96, 50, 16, 13, 7, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(43, 1, '2024-04-17', 60, 71, 10, 20, 25, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(44, 2, '2024-04-17', 10, 65, 28, 13, 3, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(45, 1, '2024-04-18', 36, 94, 3, 16, 12, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(46, 2, '2024-04-18', 56, 96, 30, 26, 0, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(47, 1, '2024-04-19', 47, 11, 13, 7, 17, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(48, 2, '2024-04-19', 80, 85, 0, 27, 10, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(49, 1, '2024-04-20', 19, 17, 21, 14, 14, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(50, 2, '2024-04-20', 68, 12, 7, 6, 4, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(51, 1, '2024-04-21', 47, 96, 21, 1, 16, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(52, 2, '2024-04-21', 74, 47, 18, 22, 26, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(53, 1, '2024-04-22', 29, 40, 11, 16, 22, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(54, 2, '2024-04-22', 64, 1, 3, 18, 17, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(55, 1, '2024-04-23', 91, 74, 19, 22, 19, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(56, 2, '2024-04-23', 58, 85, 1, 8, 18, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(57, 1, '2024-04-24', 13, 57, 22, 24, 25, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(58, 2, '2024-04-24', 34, 36, 20, 0, 5, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(59, 1, '2024-04-25', 57, 89, 30, 19, 1, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(60, 2, '2024-04-25', 42, 93, 6, 27, 20, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(61, 1, '2024-04-26', 92, 74, 30, 3, 22, '2024-04-27 07:28:57', '2024-04-27 07:28:57'),
	(62, 2, '2024-04-26', 39, 24, 6, 3, 17, '2024-04-27 07:28:57', '2024-04-27 07:28:57');

-- Dumping structure for table ai.contents
CREATE TABLE IF NOT EXISTS `contents` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `user_id` bigint unsigned NOT NULL,
  `content_type` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `content_data` json NOT NULL,
  `is_delete` tinyint(1) NOT NULL,
  `create_id` bigint unsigned NOT NULL,
  `update_id` bigint unsigned NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `contents_user_id_foreign` (`user_id`),
  KEY `contents_create_id_foreign` (`create_id`),
  KEY `contents_update_id_foreign` (`update_id`),
  CONSTRAINT `contents_create_id_foreign` FOREIGN KEY (`create_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  CONSTRAINT `contents_update_id_foreign` FOREIGN KEY (`update_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  CONSTRAINT `contents_user_id_foreign` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.contents: ~0 rows (approximately)

-- Dumping structure for table ai.content_broadcasts
CREATE TABLE IF NOT EXISTS `content_broadcasts` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `user_id` bigint unsigned NOT NULL,
  `broadcast_id` bigint unsigned NOT NULL,
  `content_id` bigint unsigned DEFAULT NULL,
  `content_sticker` json DEFAULT NULL,
  `sequence` int NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `content_broadcasts_user_id_foreign` (`user_id`),
  KEY `content_broadcasts_broadcast_id_foreign` (`broadcast_id`),
  KEY `content_broadcasts_content_id_foreign` (`content_id`),
  CONSTRAINT `content_broadcasts_broadcast_id_foreign` FOREIGN KEY (`broadcast_id`) REFERENCES `broadcasts` (`id`) ON DELETE CASCADE,
  CONSTRAINT `content_broadcasts_content_id_foreign` FOREIGN KEY (`content_id`) REFERENCES `contents` (`id`) ON DELETE CASCADE,
  CONSTRAINT `content_broadcasts_user_id_foreign` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.content_broadcasts: ~0 rows (approximately)

-- Dumping structure for table ai.content_keywords
CREATE TABLE IF NOT EXISTS `content_keywords` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `user_id` bigint unsigned NOT NULL,
  `content_id` bigint unsigned NOT NULL,
  `keyword_id` bigint unsigned NOT NULL,
  `sequence` int NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `content_keywords_user_id_foreign` (`user_id`),
  KEY `content_keywords_content_id_foreign` (`content_id`),
  KEY `content_keywords_keyword_id_foreign` (`keyword_id`),
  CONSTRAINT `content_keywords_content_id_foreign` FOREIGN KEY (`content_id`) REFERENCES `contents` (`id`) ON DELETE CASCADE,
  CONSTRAINT `content_keywords_keyword_id_foreign` FOREIGN KEY (`keyword_id`) REFERENCES `keywords` (`id`) ON DELETE CASCADE,
  CONSTRAINT `content_keywords_user_id_foreign` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.content_keywords: ~0 rows (approximately)

-- Dumping structure for table ai.failed_jobs
CREATE TABLE IF NOT EXISTS `failed_jobs` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `uuid` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `connection` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `queue` text COLLATE utf8mb4_unicode_ci NOT NULL,
  `payload` longtext COLLATE utf8mb4_unicode_ci NOT NULL,
  `exception` longtext COLLATE utf8mb4_unicode_ci NOT NULL,
  `failed_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `failed_jobs_uuid_unique` (`uuid`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.failed_jobs: ~0 rows (approximately)

-- Dumping structure for table ai.interaction_logs
CREATE TABLE IF NOT EXISTS `interaction_logs` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `user_id` bigint unsigned NOT NULL,
  `timestamp` timestamp NOT NULL,
  `action_type` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `interaction_logs_user_id_foreign` (`user_id`),
  CONSTRAINT `interaction_logs_user_id_foreign` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.interaction_logs: ~0 rows (approximately)

-- Dumping structure for table ai.jobs
CREATE TABLE IF NOT EXISTS `jobs` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `queue` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `payload` longtext COLLATE utf8mb4_unicode_ci NOT NULL,
  `attempts` tinyint unsigned NOT NULL,
  `reserved_at` int unsigned DEFAULT NULL,
  `available_at` int unsigned NOT NULL,
  `created_at` int unsigned NOT NULL,
  PRIMARY KEY (`id`),
  KEY `jobs_queue_index` (`queue`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.jobs: ~0 rows (approximately)

-- Dumping structure for table ai.keywords
CREATE TABLE IF NOT EXISTS `keywords` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `user_id` bigint unsigned NOT NULL,
  `keyword` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `is_delete` tinyint(1) NOT NULL,
  `create_id` bigint unsigned NOT NULL,
  `update_id` bigint unsigned NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `keywords_user_id_foreign` (`user_id`),
  KEY `keywords_create_id_foreign` (`create_id`),
  KEY `keywords_update_id_foreign` (`update_id`),
  CONSTRAINT `keywords_create_id_foreign` FOREIGN KEY (`create_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  CONSTRAINT `keywords_update_id_foreign` FOREIGN KEY (`update_id`) REFERENCES `users` (`id`) ON DELETE CASCADE,
  CONSTRAINT `keywords_user_id_foreign` FOREIGN KEY (`user_id`) REFERENCES `users` (`id`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.keywords: ~0 rows (approximately)

-- Dumping structure for table ai.migrations
CREATE TABLE IF NOT EXISTS `migrations` (
  `id` int unsigned NOT NULL AUTO_INCREMENT,
  `migration` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `batch` int NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=16 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.migrations: ~0 rows (approximately)
INSERT INTO `migrations` (`id`, `migration`, `batch`) VALUES
	(1, '2014_10_12_000000_create_users_table', 1),
	(2, '2014_10_12_100000_create_password_resets_table', 1),
	(3, '2019_08_19_000000_create_failed_jobs_table', 1),
	(4, '2019_12_14_000001_create_personal_access_tokens_table', 1),
	(5, '2024_02_04_062034_create_admins_table', 1),
	(6, '2024_02_04_103740_create_channels_table', 1),
	(7, '2024_02_04_103907_create_channel_statisticals_table', 1),
	(8, '2024_02_04_104001_create_interaction_logs_table', 1),
	(9, '2024_02_04_104024_create_contents_table', 1),
	(10, '2024_02_04_104049_create_broadcasts_table', 1),
	(11, '2024_02_04_104112_create_content_broadcasts_table', 1),
	(12, '2024_02_04_104140_create_keywords_table', 1),
	(13, '2024_02_04_104201_create_content_keywords_table', 1),
	(14, '2024_02_06_110318_create_jobs_table', 1),
	(15, '2024_06_10_233939_create_trackings_table', 2);

-- Dumping structure for table ai.password_resets
CREATE TABLE IF NOT EXISTS `password_resets` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `email` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `token` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `role` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  KEY `password_resets_email_index` (`email`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.password_resets: ~0 rows (approximately)

-- Dumping structure for table ai.personal_access_tokens
CREATE TABLE IF NOT EXISTS `personal_access_tokens` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `tokenable_type` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `tokenable_id` bigint unsigned NOT NULL,
  `name` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `token` varchar(64) COLLATE utf8mb4_unicode_ci NOT NULL,
  `abilities` text COLLATE utf8mb4_unicode_ci,
  `last_used_at` timestamp NULL DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `personal_access_tokens_token_unique` (`token`),
  KEY `personal_access_tokens_tokenable_type_tokenable_id_index` (`tokenable_type`,`tokenable_id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.personal_access_tokens: ~0 rows (approximately)

-- Dumping structure for table ai.trackings
CREATE TABLE IF NOT EXISTS `trackings` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `user_id` bigint unsigned NOT NULL,
  `keywords` json NOT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.trackings: ~0 rows (approximately)
INSERT INTO `trackings` (`id`, `user_id`, `keywords`, `created_at`, `updated_at`) VALUES
	(3, 1, '[{"name": "CNN", "number": 4}, {"name": "RNN", "number": 4}, {"name": "LSTM", "number": 2}, {"name": "văn mạnh", "number": 1}]', '2024-06-10 17:17:46', '2024-06-10 17:18:25'),
	(4, 2, '[{"name": "CNN", "number": 1}, {"name": "paper", "number": 1}, {"name": "measurement", "number": 1}, {"name": "gan", "number": 1}, {"name": "ck", "number": 1}, {"name": "data", "number": 1}]', '2024-06-10 17:19:05', '2024-06-11 10:57:42');

-- Dumping structure for table ai.users
CREATE TABLE IF NOT EXISTS `users` (
  `id` bigint unsigned NOT NULL AUTO_INCREMENT,
  `email` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `password` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `line_user_id` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `channel_id` bigint unsigned DEFAULT NULL,
  `role` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `is_delete` tinyint(1) NOT NULL DEFAULT '0',
  `is_block` tinyint(1) NOT NULL DEFAULT '0',
  `name` varchar(255) COLLATE utf8mb4_unicode_ci NOT NULL,
  `avatar` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `phone` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `address` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `gender` int DEFAULT NULL,
  `date_of_birth` date DEFAULT NULL,
  `token_verify_email` varchar(255) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `email_verified_at` timestamp NULL DEFAULT NULL,
  `remember_token` varchar(100) COLLATE utf8mb4_unicode_ci DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT NULL,
  `updated_at` timestamp NULL DEFAULT NULL,
  PRIMARY KEY (`id`),
  UNIQUE KEY `users_email_unique` (`email`),
  UNIQUE KEY `users_line_user_id_unique` (`line_user_id`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Dumping data for table ai.users: ~2 rows (approximately)
INSERT INTO `users` (`id`, `email`, `password`, `line_user_id`, `channel_id`, `role`, `is_delete`, `is_block`, `name`, `avatar`, `phone`, `address`, `gender`, `date_of_birth`, `token_verify_email`, `email_verified_at`, `remember_token`, `created_at`, `updated_at`) VALUES
	(1, 'nguyenvanmanh2001it1@gmail.com', '$2y$10$GZB0gCElw8pspJJIeaz9huWugffFLT3AeLZfGukl2mGbA8S8HrSuq', 'U9b60d708a68e2b81a7ff7f9c57540779', 1, 'manager', 0, 0, 'Nguyễn Văn Mạnh', 'https://res.cloudinary.com/dzve8benj/image/upload/v1714278897/avatars/q7z1snhh6tnx7sp3punu.jpg', '0971404372', 'Phú Đa - Phú Vang - TT Huế', 0, '2001-08-29', NULL, '2024-04-27 07:28:56', NULL, '2024-04-27 07:28:56', '2024-04-28 04:34:58'),
	(2, 'nguyenvanmanh.it1@yopmail.com', '$2y$10$GZB0gCElw8pspJJIeaz9huWugffFLT3AeLZfGukl2mGbA8S8HrSuq', 'U667ca434abb5753fd28330c0441c7c78', 2, 'manager', 0, 0, 'Nhật Minh', NULL, '01236000333', '120 Nguyễn Lương Bằng - Liên Chiểu - Đà Nẵng', 0, '2001-09-29', NULL, '2024-04-27 07:28:56', NULL, '2024-04-27 07:28:56', '2024-04-27 07:28:56');

/*!40103 SET TIME_ZONE=IFNULL(@OLD_TIME_ZONE, 'system') */;
/*!40101 SET SQL_MODE=IFNULL(@OLD_SQL_MODE, '') */;
/*!40014 SET FOREIGN_KEY_CHECKS=IFNULL(@OLD_FOREIGN_KEY_CHECKS, 1) */;
/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40111 SET SQL_NOTES=IFNULL(@OLD_SQL_NOTES, 1) */;
